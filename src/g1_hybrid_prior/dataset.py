
from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
from .robot_cfg import RobotCfg, load_robot_cfg
from .helpers import get_project_root, quat_normalize, quat_mul, quat_inv, quat_log, wrap_to_pi, rotate_world_to_body


class G1HybridPriorDataset(Dataset):
    def __init__(self, file_path:Path, robot:str="g1", lazy_load:bool=False, lazy_load_window:int=1000): 
        super().__init__()

        self.data = []
        self.dataset = []

        self.file_path = file_path
        self.yaml = str(get_project_root()/"config"/"robots.yaml")
        self.robot_cfg = load_robot_cfg(self.yaml, robot)
        self.lazy_load = lazy_load
        self.lazy_load_window = lazy_load_window
        self.header_rows = 0

        if self.lazy_load:
            self.num_frames = self._count_rows(self.file_path) - self.header_rows
            self.current_block_idx = None
            self.current_block_start = 0
            self.current_block_end = 0
            self._load_block(0)
        else: 
            self._load_all()
            self.num_frames = len(self.dataset)

                
    def __frame_building__(self, data):
        for row in range(data.shape[0]):
            cur = self.__split_row__(data[row], self.robot_cfg)
            if 0 < row < data.shape[0] - 1:
                nxt = self.__split_row__(data[row + 1], self.robot_cfg)
                prev = self.__split_row__(data[row - 1], self.robot_cfg)
            elif row == 0:
                nxt = self.__split_row__(data[row + 1], self.robot_cfg)
                prev = cur
            else:  # row == data.shape[0] - 1
                prev = self.__split_row__(data[row - 1], self.robot_cfg)
                nxt = cur

            root_lin_vel, root_ang_vel, joint_velocities = self.__compute_velocities__(
                nxt[0],  nxt[1],  nxt[2],   # next_root_pos, next_root_quat_wxyz, next_joints
                prev[0], prev[1], prev[2], # prev_root_pos, prev_root_quat_wxyz, prev_joints
            )

            frame = {
                "root_pos": cur[0],
                "root_quat_wxyz": cur[1],
                "joints": cur[2],
                "root_lin_vel": root_lin_vel,
                "root_ang_vel": root_ang_vel,
                "joint_vel": joint_velocities,
            }
            self.dataset.append(frame)



    def __split_row__(self, row: np.ndarray, robot_cfg: RobotCfg):
        """
        Splits a CSV row into root position, root quaternion (wxyz), and joint angles.
        """
        root = row[: robot_cfg.root_dim]
        joints = row[robot_cfg.root_dim : robot_cfg.dof + robot_cfg.root_dim]

        qx, qy, qz, qw = root[3:7]
        root_pos = torch.tensor(root[0:3], dtype=torch.float32)
        root_quat_wxyz = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)  # w,x,y,z
        joints = torch.tensor(joints, dtype=torch.float32)
        root_quat_wxyz = quat_normalize(root_quat_wxyz)

        return root_pos, root_quat_wxyz, joints

    def __compute_velocities__(self, next_root_pos, next_root_quat_wxyz, next_joints, prev_root_pos, prev_root_quat_wxyz, prev_joints): 
        """ 
        Computes root linear velocity, root angular velocity, and joint velocities using central differences.
        """
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

    def _count_rows(self, file_path: Path) -> int:
        """Count total number of lines in the CSV."""
        with open(file_path, "r") as f:
            return sum(1 for _ in f)

    def _load_all(self):
        """Non-lazy: load whole CSV and build all frames."""
        data = np.loadtxt(self.file_path, delimiter=",", skiprows=self.header_rows)
        self.data = data
        self.dataset = []
        self.__frame_building__(data)  # same logic you already have

    def _load_block(self, block_idx: int):
        """
        Lazy: load a block of at most `lazy_load_window` rows starting from
        global frame index = block_idx * lazy_load_window.
        """
        start = block_idx * self.lazy_load_window

        # Skip header rows + all previous frames
        skiprows = self.header_rows + start
        max_rows = self.lazy_load_window

        data = np.loadtxt(
            self.file_path,
            delimiter=",",
            skiprows=skiprows,
            max_rows=max_rows,
        )

        # Update block bookkeeping
        self.current_block_idx = block_idx
        self.current_block_start = start
        self.current_block_end = start + data.shape[0]

        # Rebuild dataset for this block
        self.data = data
        self.dataset = []
        self.__frame_building__(data)
        


    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        if not self.lazy_load:
            return self.dataset[idx]

        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of range (0, {self.num_frames - 1})")

        # Check if idx falls in the current block
        if not (self.current_block_start <= idx < self.current_block_end):
            block_idx = idx // self.lazy_load_window
            self._load_block(block_idx)

        local_idx = idx - self.current_block_start
        return self.dataset[local_idx]


    