
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
from .robot_cfg import RobotCfg, load_robot_cfg
from .helpers import get_project_root, quat_normalize, quat_mul, quat_inv, quat_log


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
        self.device = "cuda" if device is None and torch.cuda.is_available() else device

        self.__load_data__(file_path)

    def __load_data__(self, file_path):
        if self.lazy_load:
            self.__lazy_load_data__(file_path)
        else: 
            # Implement full data loading logic here
            data = np.loadtxt(file_path, delimiter=",")
            if data.shape[1] != self.robot_cfg.expected_cols:
                raise ValueError(
                    f"CSV has {data.shape[1]} cols, but {self.robot_cfg.name} expects "
                    f"{self.robot_cfg.expected_cols} (root {self.robot_cfg.root_dim} + dof {self.robot_cfg.dof})."
                )
            self.data = data
            for row in data:
                root_pos, root_quat_xyzw, root_quat_wxyz, joints = self.__split_row__(row, self.robot_cfg)
                frame = {
                    "root_pos": root_pos,
                    "root_quat_xyzw": root_quat_xyzw,
                    "root_quat_wxyz": root_quat_wxyz,
                    "joints": joints,
                    "root_vel": None,
                    "joint_vel": None
                }
                self.dataset.append(frame)
                
    def __lazy_load_data__(self, file_path):
        data = np.loadtxt(file_path, delimiter=",", max_rows=self.lazy_load_window, skiprows=(self.block_index * self.lazy_load_window + 1))
        self.data = data
        self.block_index += 1
        for row in range(data.shape[0]):
            root_pos, root_quat_xyzw, root_quat_wxyz, joints = self.__split_row__(data[row], self.robot_cfg)
            next_root_pos, next_root_quat_xyzw, next_root_quat_wxyz, next_joints = self.__split_row__(data[row+1], self.robot_cfg)
            root_lin_vel = root_pos
            root_ang_vel = root_quat_wxyz
            joint_velocities = joints
            if row != data.shape[0]-1 or row != 0:
                root_lin_vel, root_ang_vel, joint_velocities = self.__compute_velocities__(
                    root_pos, 
                    root_quat_wxyz, 
                    joints, 
                    next_root_pos, 
                    next_root_quat_wxyz, 
                    next_joints
                )
            frame = {
                "root_pos": root_pos,
                "root_quat_xyzw": root_quat_xyzw,
                "root_quat_wxyz": root_quat_wxyz,
                "joints": joints,
                "root_lin_vel": root_lin_vel,
                "root_ang_vel": root_ang_vel,
                "joint_vel": joint_velocities
            }
            self.dataset.append(frame)


    def __split_row__(self, row: np.ndarray, robot_cfg: RobotCfg) -> tuple:

        root = row[: robot_cfg.root_dim]
        joints = row[robot_cfg.root_dim : robot_cfg.dof + robot_cfg.root_dim]

        qx, qy, qz, qw = root[3:7]
        root_pos = root[0:3]
        root_quat_xyzw = np.array([qx, qy, qz, qw], dtype=row.dtype)  # x,y,z,w
        root_quat_wxyz = np.array([qw, qx, qy, qz], dtype=row.dtype)  # w,x,y,z

        return root_pos, root_quat_xyzw, root_quat_wxyz, joints

    def __compute_velocities__(self, root_pos, root_quat_wxyz, joints, next_root_pos, next_root_quat_wxyz, next_joints): 
        dt = 1.0 / self.robot_cfg.fps
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.lazy_load: 
            if idx > len(self.dataset):
                self.dataset = []
                self.__lazy_load_data__(self.file_path)
            block_idx = (idx - self.block_index * self.lazy_load_window)+1
            return self.dataset[block_idx]
        else:
            return self.dataset[idx]

    