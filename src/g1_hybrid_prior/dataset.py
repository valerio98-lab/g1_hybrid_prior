
from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
from .robot_cfg import RobotCfg, load_robot_cfg
from .helpers import get_project_root, quat_normalize, quat_mul, quat_inv, quat_log, wrap_to_pi, quat_rotate_inv


class G1HybridPriorDataset(Dataset):
    def __init__(self, file_path:Path, robot:str="g1", lazy_load:bool=False, lazy_load_window:int=1000, vel_mode="backward"): 
        super().__init__()

        if vel_mode not in ["backward", "central"]:
            raise ValueError(f"Invalid vel_mode '{vel_mode}'. Must be 'backward' or 'central'.")

        self.vel_mode = vel_mode
        self.data = []
        self.dataset = []

        self._ctx_left = 1 if self.vel_mode in ["backward", "central"] else 0
        self._ctx_right = 1 if self.vel_mode in ["central", "backward"] else 0

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
            self._loaded_start = 0
            self._loaded_end = 0
            self._load_block(0)
        else: 
            self._load_all()
            self.num_frames = len(self.dataset)

                
    def __frame_building__(self, data):
        dt = 1.0 / self.robot_cfg.fps
        for row in range(data.shape[0]):
            cur = self.__split_row__(data[row], self.robot_cfg)
            if data.shape[0] == 1:
                prev = cur
                nxt = cur
            else: 
                if row==0:
                    prev = cur
                    nxt = self.__split_row__(data[row + 1], self.robot_cfg) if data.shape[0] > 1 else cur
                elif row == data.shape[0] - 1:
                    prev = self.__split_row__(data[row - 1], self.robot_cfg)
                    nxt = cur
                else:
                    prev = self.__split_row__(data[row - 1], self.robot_cfg)
                    nxt = self.__split_row__(data[row + 1], self.robot_cfg)

            if self.vel_mode == "central":
                if row==0: 
                    root_lin_vel, root_ang_vel, joint_velocities = self.__compute_velocities_forward__(
                        cur_root_pos=cur[0],
                        cur_root_quat_wxyz=cur[1],
                        cur_joints=cur[2],
                        next_root_pos=nxt[0],
                        next_root_quat_wxyz=nxt[1],
                        next_joints=nxt[2],
                        dt=dt,
                    )
                elif row == data.shape[0] - 1:
                    root_lin_vel, root_ang_vel, joint_velocities = self.__compute_velocities_backward__(
                        prev_root_pos=prev[0],
                        prev_root_quat_wxyz=prev[1],
                        prev_joints=prev[2],
                        cur_root_pos=cur[0],
                        cur_root_quat_wxyz=cur[1],
                        cur_joints=cur[2],
                        dt=dt,
                    )
                else:
                    root_lin_vel, root_ang_vel, joint_velocities = self.__compute_velocities_central__(
                        prev_root_pos=prev[0],
                        prev_root_quat_wxyz=prev[1],
                        prev_joints=prev[2],
                        next_root_pos=nxt[0],
                        next_root_quat_wxyz=nxt[1],
                        next_joints=nxt[2],
                        dt=dt,
                    )
            elif self.vel_mode == "backward":
                if row == 0:
                    # at first frame, use forward difference
                    root_lin_vel, root_ang_vel, joint_velocities = self.__compute_velocities_forward__(
                        cur_root_pos=cur[0],
                        cur_root_quat_wxyz=cur[1],
                        cur_joints=cur[2],
                        next_root_pos=nxt[0],
                        next_root_quat_wxyz=nxt[1],
                        next_joints=nxt[2],
                        dt=dt,
                    )
                else:
                    root_lin_vel, root_ang_vel, joint_velocities = self.__compute_velocities_backward__(
                        prev_root_pos=prev[0],
                        prev_root_quat_wxyz=prev[1],
                        prev_joints=prev[2],
                        cur_root_pos=cur[0],
                        cur_root_quat_wxyz=cur[1],
                        cur_joints=cur[2],
                        dt=dt,
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

    def __compute_velocities_forward__(
        self,
        cur_root_pos: torch.Tensor,
        cur_root_quat_wxyz: torch.Tensor,
        cur_joints: torch.Tensor,
        next_root_pos: torch.Tensor,
        next_root_quat_wxyz: torch.Tensor,
        next_joints: torch.Tensor,
        dt: float,
    ):
        """
        Forward differences (used only at boundary when vel_mode='central'):
          v ≈ (x_{t+1} - x_t) / dt
          ω ≈ log( inv(q_t) * q_{t+1} ) / dt
          q̇_joints ≈ wrap(x_{t+1} - x_t) / dt

        Velocities expressed in BODY using the CURRENT quaternion (t).
        """
        # Linear vel in world then rotate to BODY(cur)
        v_world = (next_root_pos - cur_root_pos) / dt
        q_cur = quat_normalize(cur_root_quat_wxyz)
        q_next_norm = quat_normalize(next_root_quat_wxyz)
        v_body = quat_rotate_inv(q_cur, v_world)

        # Angular vel relative rotation cur->next, expressed in cur frame
        q_rel = quat_mul(quat_inv(q_cur), q_next_norm)
        q_rel = quat_normalize(q_rel)
        w_body = quat_log(q_rel) / dt

        joint_vel = wrap_to_pi(next_joints - cur_joints) / dt

        return v_body, w_body, joint_vel

    def __compute_velocities_central__(
        self,
        prev_root_pos: torch.Tensor,
        prev_root_quat_wxyz: torch.Tensor,
        prev_joints: torch.Tensor,
        next_root_pos: torch.Tensor,
        next_root_quat_wxyz: torch.Tensor,
        next_joints: torch.Tensor,
        dt: float,
    ):
        """
        Central differences:
          v ≈ (x_{t+1} - x_{t-1}) / (2 dt)
          ω ≈ log( inv(q_{t-1}) * q_{t+1} ) / (2 dt)
          q̇ for joints ≈ wrap(x_{t+1} - x_{t-1}) / (2 dt)

        Velocities are expressed in BODY using the PREV quaternion (t-1)
        """
        denom = 2.0 * dt

        # Linear vel in world_frame then rotate to body_frame(prev)
        v_world = (next_root_pos - prev_root_pos) / denom
        q_prev_norm = quat_normalize(prev_root_quat_wxyz)
        q_next_norm = quat_normalize(next_root_quat_wxyz)
        v_body = quat_rotate_inv(q_prev_norm, v_world)

        # Angular vel: relative rotation prev->next, expressed in prev frame
        q_rel = quat_mul(quat_inv(q_prev_norm), q_next_norm)
        q_rel = quat_normalize(q_rel)
        w_body = quat_log(q_rel) / denom  

        joint_vel = wrap_to_pi(next_joints - prev_joints) / denom

        return v_body, w_body, joint_vel

    def __compute_velocities_backward__(
        self,
        prev_root_pos: torch.Tensor,
        prev_root_quat_wxyz: torch.Tensor,
        prev_joints: torch.Tensor,
        cur_root_pos: torch.Tensor,
        cur_root_quat_wxyz: torch.Tensor,
        cur_joints: torch.Tensor,
        dt: float,
    ):
        """
        Backward differences:
          v ≈ (x_t - x_{t-1}) / dt
          ω ≈ log( inv(q_{t-1}) * q_t ) / dt
          q̇ for joints ≈ wrap(x_t - x_{t-1}) / dt

        This matches what simulation does. Central differences are smoother but in simulation we only have access to 
        past and current states. Velocities are expressed in BODY(prev).
        """
        # Linear vel in world_frame then rotate to body_frame(prev)
        v_world = (cur_root_pos - prev_root_pos) / dt
        q_prev_norm = quat_normalize(prev_root_quat_wxyz)
        q_cur = quat_normalize(cur_root_quat_wxyz)
        v_body = quat_rotate_inv(q_prev_norm, v_world)

        # Angular vel: relative rotation prev->cur, expressed in prev frame
        q_rel = quat_mul(quat_inv(q_prev_norm), q_cur)
        q_rel = quat_normalize(q_rel)
        w_body = quat_log(q_rel) / dt

        # Joint vel
        joint_vel = wrap_to_pi(cur_joints - prev_joints) / dt

        return v_body, w_body, joint_vel

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
        """Lazy-load: load a block of rows from the CSV and build frames."""
        core_start = block_idx * self.lazy_load_window
        core_end = min(core_start + self.lazy_load_window, self.num_frames)
        load_start = max(0, core_start - self._ctx_left)
        load_end = min(self.num_frames, core_end + self._ctx_right)

        skiprows = self.header_rows + load_start
        max_rows = int(load_end - load_start)

        data = np.loadtxt(self.file_path, delimiter=",", skiprows=skiprows, max_rows=max_rows)
        if data.ndim == 1:
            data = data[None, :]  # Convert to 2D array with one row

        self.current_block_idx = block_idx
        self.current_block_start = core_start
        self.current_block_end = core_end
        self._loaded_start = load_start
        self._loaded_end = load_end
        self.data = data
        self.dataset = []
        self.__frame_building__(data)

        core_offset = core_start - load_start
        core_len = core_end - core_start
        self.dataset = self.dataset[core_offset : core_offset + core_len]
        

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

        local_idx = int(idx - self.current_block_start)
        return self.dataset[local_idx]



# if __name__ == "__main__":
#     from torch.utils.data import DataLoader

#     csv_path = Path("data_raw/LAFAN1_Retargeting_Dataset/g1/dance1_subject1.csv")

#     dataset = G1HybridPriorDataset(
#         file_path=csv_path,
#         robot="g1",
#         lazy_load=False,          # or False to compare behaviours
#         lazy_load_window=1000,
#     )

#     loader = DataLoader(
#         dataset,
#         batch_size=32,
#         shuffle=True,            # works: it just triggers different blocks to be loaded
#         num_workers=0,           # start with 0 to debug more easily
#         pin_memory=False,        # True later if you move batches to GPU
#     )

#     i = 0
#     for batch in loader:
#         if i == 0:
#             print(batch.keys())
#             i += 1
        
#         batch = {k: v.to("cuda") for k, v in batch.items()}
#         # dict_keys(['root_pos', 'root_quat_wxyz', 'joints',
#         #            'root_lin_vel', 'root_ang_vel', 'joint_vel'])
#         print(f"{batch['root_pos'].shape=}")        # torch.Size([32, 3])
#         print(f"{batch['root_quat_wxyz'].shape=}")  # torch.Size([32, 4])
#         print(f"{batch['joints'].shape=}")          # torch.Size([32, dof])
#         print(f"{batch['root_lin_vel'].shape=}")   # torch.Size([32, 3])
#         print(f"{batch['root_ang_vel'].shape=}")   # torch.Size([32,
#         print(f"{batch['joint_vel'].shape=}")      # torch.Size([32, dof])
#         i += 1
#         if i == 5:
#             break






    