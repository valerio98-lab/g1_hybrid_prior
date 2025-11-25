
from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
from .robot_cfg import RobotCfg, load_robot_cfg
from .helpers import get_project_root, quat_normalize, quat_mul, quat_inv, quat_log, wrap_to_pi, rotate_world_to_body


class G1HybridPriorDataset(Dataset):
    def __init__(self, file_path:Path, robot:str="g1", lazy_load:bool=False, lazy_load_window:int=1000, device=None): 
        super().__init__()

        self.data = []
        self.dataset = []

        self.file_path = file_path
        self.yaml = str(get_project_root()/"config"/"robots.yaml")
        self.robot_cfg = load_robot_cfg(self.yaml, robot)
        self.lazy_load = lazy_load
        self.lazy_load_window = lazy_load_window
        self.block_index = 0
        self.device = "cuda" if (device is None and torch.cuda.is_available()) else device or "cpu"

        self.__load_data__(file_path)

    def __load_data__(self, file_path):
        if self.lazy_load:
            data = np.loadtxt(file_path, delimiter=",", max_rows=self.lazy_load_window, skiprows=(self.block_index * self.lazy_load_window + 1))
            self.block_index += 1
        
        else: 
            data = np.loadtxt(file_path, delimiter=",")
        
        if data.shape[1] != self.robot_cfg.expected_cols:
            raise ValueError(
                f"CSV has {data.shape[1]} cols, but {self.robot_cfg.name} expects "
                f"{self.robot_cfg.expected_cols} (root {self.robot_cfg.root_dim} + dof {self.robot_cfg.dof})."
            )
        self.data = data
        
        self.__frame_building__(file_path, data)

                
    def __frame_building__(self, file_path, data):
        for row in range(self.data.shape[0]):
            cur = self.__split_row__(data[row], self.robot_cfg)
            if 0 < row < data.shape[0]-1:
                nxt = self.__split_row__(data[row+1], self.robot_cfg)
                prev = self.__split_row__(data[row-1], self.robot_cfg)
            elif row == 0:
                nxt = self.__split_row__(data[row+1], self.robot_cfg)
                prev = cur  
            else:  # row == data.shape[0]-1
                prev = self.__split_row__(data[row-1], self.robot_cfg)
                nxt = cur 

            root_lin_vel, root_ang_vel, joint_velocities = self.__compute_velocities__(
                nxt[0], nxt[1], nxt[2],   # next_root_pos, next_root_quat_wxyz, next_joints
                prev[0], prev[1], prev[2],   # prev_root_pos, prev_root_quat_wxyz, prev_joints
            )

            frame = {
                "root_pos": cur[0],
                "root_quat_wxyz": cur[1],
                "joints": cur[2],
                "root_lin_vel": root_lin_vel,
                "root_ang_vel": root_ang_vel,
                "joint_vel": joint_velocities
            }
            self.dataset.append(frame)


    def __split_row__(self, row: np.ndarray, robot_cfg: RobotCfg) -> tuple:
        root = row[: robot_cfg.root_dim]
        joints = row[robot_cfg.root_dim : robot_cfg.dof + robot_cfg.root_dim]

        qx, qy, qz, qw = root[3:7]
        root_pos = torch.tensor(root[0:3], dtype=torch.float32)
        root_quat_wxyz = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)  # w,x,y,z
        joints = torch.tensor(joints, dtype=torch.float32)
        root_quat_wxyz = quat_normalize(root_quat_wxyz)

        return root_pos, root_quat_wxyz, joints

    def __compute_velocities__(self, next_root_pos, next_root_quat_wxyz, next_joints, prev_root_pos, prev_root_quat_wxyz, prev_joints): 
        dt = 1.0 / self.robot_cfg.fps
        root_lin_vel = (next_root_pos - prev_root_pos) / (2 * dt) ## Central difference
        q_prev_inv = quat_inv(prev_root_quat_wxyz)
        q_rel = quat_mul(q_prev_inv, next_root_quat_wxyz)
        q_rel = quat_normalize(q_rel)
        q_prev_norm = quat_normalize(prev_root_quat_wxyz)
        root_lin_vel = rotate_world_to_body(root_lin_vel, q_prev_norm)
        root_ang_vel = quat_log(q_rel) / (2 * dt)
        joint_velocities = wrap_to_pi(next_joints - prev_joints) / (2 * dt)
        return root_lin_vel, root_ang_vel, joint_velocities
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.lazy_load: 
            if idx > len(self.dataset):
                self.dataset = []
                self.__load_data__(self.file_path)
            block_idx = (idx - self.block_index * self.lazy_load_window)+1
            return self.dataset[block_idx]
        else:
            return self.dataset[idx]

    