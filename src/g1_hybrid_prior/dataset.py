from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from .robot_cfg import RobotCfg, load_robot_cfg
from .helpers import (
    get_project_root,
    quat_normalize,
    quat_mul,
    quat_inv,
    quat_log,
    wrap_to_pi,
    quat_rotate_inv,
)


class G1HybridPriorDataset(Dataset):
    def __init__(
        self,
        file_path: Path | str,
        robot: str = "g1",
        lazy_load: bool = False,
        lazy_load_window: int = 1000,
        vel_mode="central",
        dataset_type: str = "raw",
    ):
        super().__init__()

        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.file_path}")

        if self.file_path.suffix not in [".csv"] and not self.file_path.is_dir():
            raise ValueError(
                f"Invalid file type: {self.file_path.suffix}. Must be a CSV file or a directory."
            )

        if vel_mode not in ["backward", "central"]:
            raise ValueError(
                f"Invalid vel_mode '{vel_mode}'. Must be 'backward' or 'central'."
            )

        if dataset_type not in ["raw", "augmented"]:
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'. Must be 'raw' or 'augmented'."
            )

        self.data = []
        self.dataset = []
        self.vel_mode = vel_mode
        self.dataset_type = dataset_type
        self._ctx_left = 1 if self.vel_mode in ["backward", "central"] else 0
        self._ctx_right = 1 if self.vel_mode in ["central", "backward"] else 0

        self.yaml = str(get_project_root() / "config" / "robots.yaml")
        self.robot_cfg = load_robot_cfg(self.yaml, robot)
        self.lazy_load = lazy_load
        self.lazy_load_window = lazy_load_window
        self.header_rows = 0

        self.base_cols = self.robot_cfg.expected_cols
        self.ee_dim = 0

        if self.dataset_type == "augmented":
            self.ee_dim = self.robot_cfg.num_ee * 3
            self.total_expected_cols = self.base_cols + self.ee_dim
        else:
            self.total_expected_cols = self.base_cols

        if self.file_path.is_dir():
            if lazy_load:
                print(
                    "[Dataset] WARNING: Lazy load not supported for directories. Switching to full load."
                )
                lazy_load = False

            files = sorted(list(self.file_path.glob("*.csv")))
            print(
                f"[Dataset] Loading dataset from directory: {self.file_path}, {len(files)} files found."
            )

            total_frames = 0
            for f in tqdm(files, desc="Loading dataset files"):
                file_frames = self._load_file(f)
                self.dataset.extend(file_frames)
                total_frames += len(file_frames)
            print(f"[Dataset] Total frames loaded: {total_frames}")
            self.num_frames = len(self.dataset)

        elif self.file_path.is_file():
            if lazy_load:
                self.lazy_load = True
                self.lazy_load_window = lazy_load_window
                self.header_rows = 0
                self.num_frames = self._count_rows(self.file_path) - self.header_rows
                self.current_block_idx = None
                self._load_block(0)
            else:
                self.lazy_load = False
                # ORA FUNZIONA: _load_file esiste
                self.dataset = self._load_file(self.file_path)
                self.num_frames = len(self.dataset)

        # Gestione caso file singolo lazy (già gestito sopra nell'if is_file, ma teniamo per sicurezza logica)
        # Nota: ho rimosso la duplicazione di logica che avevi in fondo all'init precedente

    def _load_file(self, path: Path):
        """Helper per caricare un singolo file CSV e restituire la lista di frame."""
        # Usa header=None perché i tuoi file augmented non hanno header
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=self.header_rows)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []

        if data.ndim == 1:
            data = data[None, :]

        # Chiama frame_building che ora RESTITUISCE la lista
        return self.__frame_building__(data)

    def __frame_building__(self, data):
        """
        ORA RESTITUISCE UNA LISTA DI FRAME invece di appendere a self.dataset.
        Questo rende la funzione 'pura' e usabile nei loop.
        """
        dt = 1.0 / self.robot_cfg.fps

        if data.shape[1] != self.total_expected_cols:
            raise ValueError(
                f"File mismatch: expected {self.total_expected_cols} columns, found {data.shape[1]}"
            )

        frames_list = []  # Buffer locale

        for row in range(data.shape[0]):
            cur = self.__split_row__(data[row], self.robot_cfg)

            if data.shape[0] == 1:
                prev = cur
                nxt = cur
            else:
                if row == 0:
                    prev = cur
                    nxt = (
                        self.__split_row__(data[row + 1], self.robot_cfg)
                        if data.shape[0] > 1
                        else cur
                    )
                elif row == data.shape[0] - 1:
                    prev = self.__split_row__(data[row - 1], self.robot_cfg)
                    nxt = cur
                else:
                    prev = self.__split_row__(data[row - 1], self.robot_cfg)
                    nxt = self.__split_row__(data[row + 1], self.robot_cfg)

            if self.vel_mode == "central":
                if row == 0:
                    root_lin_vel, root_ang_vel, joint_velocities = (
                        self.__compute_velocities_forward__(
                            cur_root_pos=cur[0],
                            cur_root_quat_wxyz=cur[1],
                            cur_joints=cur[2],
                            next_root_pos=nxt[0],
                            next_root_quat_wxyz=nxt[1],
                            next_joints=nxt[2],
                            dt=dt,
                        )
                    )
                elif row == data.shape[0] - 1:
                    root_lin_vel, root_ang_vel, joint_velocities = (
                        self.__compute_velocities_backward__(
                            prev_root_pos=prev[0],
                            prev_root_quat_wxyz=prev[1],
                            prev_joints=prev[2],
                            cur_root_pos=cur[0],
                            cur_root_quat_wxyz=cur[1],
                            cur_joints=cur[2],
                            dt=dt,
                        )
                    )
                else:
                    root_lin_vel, root_ang_vel, joint_velocities = (
                        self.__compute_velocities_central__(
                            prev_root_pos=prev[0],
                            prev_root_quat_wxyz=prev[1],
                            prev_joints=prev[2],
                            cur_root_quat_wxyz=cur[1],
                            next_root_pos=nxt[0],
                            next_root_quat_wxyz=nxt[1],
                            next_joints=nxt[2],
                            dt=dt,
                        )
                    )
            elif self.vel_mode == "backward":
                if row == 0:
                    root_lin_vel, root_ang_vel, joint_velocities = (
                        self.__compute_velocities_forward__(
                            cur_root_pos=cur[0],
                            cur_root_quat_wxyz=cur[1],
                            cur_joints=cur[2],
                            next_root_pos=nxt[0],
                            next_root_quat_wxyz=nxt[1],
                            next_joints=nxt[2],
                            dt=dt,
                        )
                    )
                else:
                    root_lin_vel, root_ang_vel, joint_velocities = (
                        self.__compute_velocities_backward__(
                            prev_root_pos=prev[0],
                            prev_root_quat_wxyz=prev[1],
                            prev_joints=prev[2],
                            cur_root_pos=cur[0],
                            cur_root_quat_wxyz=cur[1],
                            cur_joints=cur[2],
                            dt=dt,
                        )
                    )

            frame = {
                "root_pos": cur[0],
                "root_quat_wxyz": cur[1],
                "joints": cur[2],
                "root_lin_vel": root_lin_vel,
                "root_ang_vel": root_ang_vel,
                "joint_vel": joint_velocities,
            }

            if cur[3] is not None:
                frame["ee_pos"] = cur[3]

            frames_list.append(frame)

        return frames_list  # Ritorna la lista!

    def __split_row__(self, row: np.ndarray, robot_cfg: RobotCfg):
        root_end = robot_cfg.root_dim
        joints_end = root_end + robot_cfg.dof

        root = row[:root_end]
        joints = row[root_end:joints_end]

        qx, qy, qz, qw = root[3:7]
        root_pos = torch.tensor(root[0:3], dtype=torch.float32)
        root_quat_wxyz = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
        joints_t = torch.tensor(joints, dtype=torch.float32)
        root_quat_wxyz = quat_normalize(root_quat_wxyz)

        ee_pos_t = None
        if self.dataset_type == "augmented":
            ee_flat = row[joints_end : joints_end + self.ee_dim]
            ee_pos_t = torch.tensor(ee_flat, dtype=torch.float32).view(
                robot_cfg.num_ee, 3
            )

        return root_pos, root_quat_wxyz, joints_t, ee_pos_t

    def __compute_velocities_forward__(
        self,
        cur_root_pos,
        cur_root_quat_wxyz,
        cur_joints,
        next_root_pos,
        next_root_quat_wxyz,
        next_joints,
        dt,
    ):
        v_world = (next_root_pos - cur_root_pos) / dt
        q_cur = quat_normalize(cur_root_quat_wxyz)
        q_next_norm = quat_normalize(next_root_quat_wxyz)
        v_body = quat_rotate_inv(q_cur, v_world)

        q_rel = quat_mul(quat_inv(q_cur), q_next_norm)
        q_rel = quat_normalize(q_rel)
        w_body = quat_log(q_rel) / dt

        joint_vel = wrap_to_pi(next_joints - cur_joints) / dt
        return v_body, w_body, joint_vel

    def __compute_velocities_central__(
        self,
        prev_root_pos,
        prev_root_quat_wxyz,
        prev_joints,
        cur_root_quat_wxyz,
        next_root_pos,
        next_root_quat_wxyz,
        next_joints,
        dt,
    ):
        denom = 2.0 * dt
        v_world = (next_root_pos - prev_root_pos) / denom

        q_cur_norm = quat_normalize(cur_root_quat_wxyz)
        v_body = quat_rotate_inv(q_cur_norm, v_world)

        q_prev_norm = quat_normalize(prev_root_quat_wxyz)
        q_next_norm = quat_normalize(next_root_quat_wxyz)

        q_rel = quat_mul(quat_inv(q_prev_norm), q_next_norm)
        q_rel = quat_normalize(q_rel)
        w_body = quat_log(q_rel) / denom

        joint_vel = wrap_to_pi(next_joints - prev_joints) / denom
        return v_body, w_body, joint_vel

    def __compute_velocities_backward__(
        self,
        prev_root_pos,
        prev_root_quat_wxyz,
        prev_joints,
        cur_root_pos,
        cur_root_quat_wxyz,
        cur_joints,
        dt,
    ):
        v_world = (cur_root_pos - prev_root_pos) / dt
        q_cur = quat_normalize(cur_root_quat_wxyz)
        v_body = quat_rotate_inv(q_cur, v_world)

        q_rel = quat_mul(quat_inv(prev_root_quat_wxyz), q_cur)
        q_rel = quat_normalize(q_rel)
        w_body = quat_log(q_rel) / dt

        joint_vel = wrap_to_pi(cur_joints - prev_joints) / dt
        return v_body, w_body, joint_vel

    def _count_rows(self, file_path: Path) -> int:
        with open(file_path, "r") as f:
            return sum(1 for _ in f)

    def _load_all(self):
        data = np.loadtxt(self.file_path, delimiter=",", skiprows=self.header_rows)
        if data.ndim == 1:
            data = data[None, :]
        self.data = data
        # Ora __frame_building__ ritorna la lista, quindi assegniamo
        self.dataset = self.__frame_building__(data)

    def _load_block(self, block_idx: int):
        core_start = block_idx * self.lazy_load_window
        core_end = min(core_start + self.lazy_load_window, self.num_frames)
        load_start = max(0, core_start - self._ctx_left)
        load_end = min(self.num_frames, core_end + self._ctx_right)

        skiprows = self.header_rows + load_start
        max_rows = int(load_end - load_start)

        data = np.loadtxt(
            self.file_path, delimiter=",", skiprows=skiprows, max_rows=max_rows
        )
        if data.ndim == 1:
            data = data[None, :]

        self.current_block_idx = block_idx
        self.current_block_start = core_start
        self.current_block_end = core_end
        self._loaded_start = load_start
        self._loaded_end = load_end
        self.data = data
        # Assegnazione corretta
        full_block = self.__frame_building__(data)

        core_offset = core_start - load_start
        core_len = core_end - core_start
        self.dataset = full_block[core_offset : core_offset + core_len]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        if not self.lazy_load:
            return self.dataset[idx]

        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of range (0, {self.num_frames - 1})")

        if not (self.current_block_start <= idx < self.current_block_end):
            block_idx = idx // self.lazy_load_window
            self._load_block(block_idx)

        local_idx = int(idx - self.current_block_start)
        return self.dataset[local_idx]
