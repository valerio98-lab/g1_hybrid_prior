from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

from .robot_cfg import RobotCfg, load_robot_cfg
from ..helpers import (
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
        dataset_type: str = "raw",  # "raw" | "augmented" (CSV legacy). In NPZ, EE/body are read if present.
        input_format: str = "auto",  # "auto" | "csv" | "npz"
    ):
        super().__init__()

        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.file_path}")

        if vel_mode not in ["backward", "central"]:
            raise ValueError(
                f"Invalid vel_mode '{vel_mode}'. Must be 'backward' or 'central'."
            )

        if dataset_type not in ["raw", "augmented"]:
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'. Must be 'raw' or 'augmented'."
            )

        if input_format not in ["auto", "csv", "npz"]:
            raise ValueError(
                f"Invalid input_format '{input_format}'. Must be 'auto', 'csv', or 'npz'."
            )

        self._body_names = None
        self._ee_names = None

        self.vel_mode = vel_mode
        self.dataset_type = dataset_type
        self._ctx_left = 1 if self.vel_mode in ["backward", "central"] else 0
        self._ctx_right = 1 if self.vel_mode in ["central", "backward"] else 0

        self.yaml = str(get_project_root() / "config" / "robots.yaml")
        self.robot_cfg = load_robot_cfg(self.yaml, robot)

        self.lazy_load = lazy_load
        self.lazy_load_window = lazy_load_window
        self.header_rows = 0

        # CSV expected dims (legacy)
        self.base_cols = self.robot_cfg.expected_cols
        self.ee_dim_csv = 0
        if self.dataset_type == "augmented":
            self.ee_dim_csv = self.robot_cfg.num_ee * 3
            self.total_expected_cols = self.base_cols + self.ee_dim_csv
        else:
            self.total_expected_cols = self.base_cols

        # Decide format
        self.input_format = self._infer_format(self.file_path, input_format)

        # Internal storage
        self.dataset = []
        self.num_frames = 0

        # NPZ mode: no lazy-load
        if self.input_format == "npz" and self.lazy_load:
            print(
                "[Dataset] WARNING: Lazy load not supported for NPZ. Switching to full load."
            )
            self.lazy_load = False

        # Load
        if self.file_path.is_dir():
            if self.input_format == "csv":
                self._load_directory_csv(self.file_path)
            else:
                self._load_directory_npz(self.file_path)

        elif self.file_path.is_file():
            if self.input_format == "csv":
                self._load_single_csv(self.file_path)
            else:
                self._load_single_npz(self.file_path)

    # ----------------------------
    # Format detection
    # ----------------------------
    def _infer_format(self, path: Path, input_format: str) -> str:
        if input_format in ["csv", "npz"]:
            return input_format

        # auto
        if path.is_file():
            if path.suffix == ".npz":
                return "npz"
            if path.suffix == ".csv":
                return "csv"
            raise ValueError(f"Invalid file type: {path.suffix}. Must be .csv or .npz")

        if path.is_dir():
            # decide based on what it contains
            npz_files = list(path.glob("*.npz"))
            csv_files = list(path.glob("*.csv"))
            if len(npz_files) > 0 and len(csv_files) == 0:
                return "npz"
            if len(csv_files) > 0 and len(npz_files) == 0:
                return "csv"
            if len(npz_files) > 0 and len(csv_files) > 0:
                # ambiguous: default NPZ because it's "new flow"
                print(
                    "[Dataset] WARNING: directory contains both .csv and .npz. Defaulting to NPZ."
                )
                return "npz"
            raise ValueError(f"Directory has no .csv or .npz files: {path}")

        raise ValueError(f"Unsupported path: {path}")

    # ----------------------------
    # CSV loading (legacy)
    # ----------------------------
    def _load_directory_csv(self, directory: Path):
        if self.lazy_load:
            print(
                "[Dataset] WARNING: Lazy load not supported for directories. Switching to full load."
            )
            self.lazy_load = False

        files = sorted(list(directory.glob("*.csv")))
        if len(files) == 0:
            raise FileNotFoundError(f"No .csv files found in directory: {directory}")

        print(
            f"[Dataset] Loading CSV dataset from directory: {directory}, {len(files)} files found."
        )

        total_frames = 0
        for f in tqdm(files, desc="Loading dataset files"):
            file_frames = self._load_file_csv(f)
            self.dataset.extend(file_frames)
            total_frames += len(file_frames)

        self.num_frames = len(self.dataset)
        print(f"[Dataset] Total frames loaded: {self.num_frames}")

    def _load_single_csv(self, path: Path):
        if self.lazy_load:
            self.header_rows = 0
            self.num_frames = self._count_rows(path) - self.header_rows
            self.current_block_idx = None
            self._load_block_csv(0)
        else:
            self.dataset = self._load_file_csv(path)
            self.num_frames = len(self.dataset)

    def _load_file_csv(self, path: Path):
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=self.header_rows)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []

        if data.ndim == 1:
            data = data[None, :]

        return self.__frame_building_from_csv_rows__(data)

    def __frame_building_from_csv_rows__(self, data: np.ndarray):
        dt = 1.0 / self.robot_cfg.fps

        if data.shape[1] != self.total_expected_cols:
            raise ValueError(
                f"File mismatch: expected {self.total_expected_cols} columns, found {data.shape[1]}"
            )

        frames_list = []
        for row in range(data.shape[0]):
            cur = self.__split_row_csv__(data[row], self.robot_cfg)

            if data.shape[0] == 1:
                prev = cur
                nxt = cur
            else:
                if row == 0:
                    prev = cur
                    nxt = self.__split_row_csv__(data[row + 1], self.robot_cfg)
                elif row == data.shape[0] - 1:
                    prev = self.__split_row_csv__(data[row - 1], self.robot_cfg)
                    nxt = cur
                else:
                    prev = self.__split_row_csv__(data[row - 1], self.robot_cfg)
                    nxt = self.__split_row_csv__(data[row + 1], self.robot_cfg)

            root_lin_vel, root_ang_vel, joint_velocities = self.__compute_velocities__(
                prev=prev,
                cur=cur,
                nxt=nxt,
                dt=dt,
                row=row,
                n_rows=data.shape[0],
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

        return frames_list

    def __split_row_csv__(self, row: np.ndarray, robot_cfg: RobotCfg):
        root_end = robot_cfg.root_dim
        joints_end = root_end + robot_cfg.dof

        root = row[:root_end]
        joints = row[root_end:joints_end]

        # CSV root format: [x y z qx qy qz qw]
        qx, qy, qz, qw = root[3:7]
        root_pos = torch.tensor(root[0:3], dtype=torch.float32)
        root_quat_wxyz = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
        root_quat_wxyz = quat_normalize(root_quat_wxyz)

        joints_t = torch.tensor(joints, dtype=torch.float32)

        ee_pos_t = None
        if self.dataset_type == "augmented":
            ee_flat = row[joints_end : joints_end + self.ee_dim_csv]
            ee_pos_t = torch.tensor(ee_flat, dtype=torch.float32).view(
                robot_cfg.num_ee, 3
            )

        return root_pos, root_quat_wxyz, joints_t, ee_pos_t

    def _load_directory_npz(self, directory: Path):
        files = sorted(list(directory.glob("*.npz")))
        if len(files) == 0:
            raise FileNotFoundError(f"No .npz files found in directory: {directory}")

        print(
            f"[Dataset] Loading NPZ dataset from directory: {directory}, {len(files)} files found."
        )

        total_frames = 0
        for f in tqdm(files, desc="Loading NPZ files"):
            file_frames = self._load_file_npz(f)
            self.dataset.extend(file_frames)
            total_frames += len(file_frames)

        self.num_frames = len(self.dataset)
        print(f"[Dataset] Total frames loaded: {self.num_frames}")

    def _load_single_npz(self, path: Path):
        self.dataset = self._load_file_npz(path)
        self.num_frames = len(self.dataset)

    def _load_file_npz(self, path: Path):
        """
        Expected NPZ keys (your new flow):
          - root: (T,7) [x y z qx qy qz qw]
          - q_joints_yaml: (T,dof) joints in robots.yaml order
          - body_pos: (T,K,3) optional
          - body_names: (K,) optional
          - ee_pos: (T,E,3) optional
          - ee_names: (E,) optional
        """
        try:
            npz = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []

        if "root" not in npz or "q_joints_yaml" not in npz:
            raise ValueError(
                f"NPZ {path.name} missing required keys. Need at least 'root' and 'q_joints_yaml'. "
                f"Found: {list(npz.keys())}"
            )

        root = npz["root"]  # (T,7) float32
        joints = npz["q_joints_yaml"]  # (T,dof) float32

        if root.ndim != 2 or root.shape[1] != 7:
            raise ValueError(f"NPZ {path.name}: 'root' must be (T,7). Got {root.shape}")
        if joints.ndim != 2 or joints.shape[1] != self.robot_cfg.dof:
            raise ValueError(
                f"NPZ {path.name}: 'q_joints_yaml' must be (T,{self.robot_cfg.dof}). Got {joints.shape}"
            )

        body_pos = npz["body_pos"] if "body_pos" in npz else None
        body_names = npz["body_names"] if "body_names" in npz else None

        ee_pos = npz["ee_pos"] if "ee_pos" in npz else None
        ee_names = npz["ee_names"] if "ee_names" in npz else None

        if body_names is not None:
            bn = [str(x) for x in body_names.tolist()]  # robust
            if self._body_names is None:
                self._body_names = bn
            else:
                if self._body_names != bn:
                    raise ValueError(f"Inconsistent body_names in {path.name}")

        if ee_names is not None:
            en = [str(x) for x in ee_names.tolist()]
            if self._ee_names is None:
                self._ee_names = en
            else:
                if self._ee_names != en:
                    raise ValueError(f"Inconsistent ee_names in {path.name}")

        return self.__frame_building_from_npz_arrays__(
            root=root,
            joints=joints,
            ee_pos=ee_pos,
            body_pos=body_pos,
        )

    def get_body_names(self):
        return self._body_names

    def get_ee_names(self):
        return self._ee_names

    def body_indices(self, names):
        if self._body_names is None:
            raise RuntimeError("Dataset has no body_names (no body_pos in NPZ).")
        name_to_i = {n: i for i, n in enumerate(self._body_names)}
        missing = [n for n in names if n not in name_to_i]
        if missing:
            raise ValueError(f"Requested body names not in dataset: {missing}")
        return torch.tensor([name_to_i[n] for n in names], dtype=torch.long)

    def ee_indices(self, names):
        if self._ee_names is None:
            raise RuntimeError("Dataset has no ee_names (no ee_pos in NPZ).")
        name_to_i = {n: i for i, n in enumerate(self._ee_names)}
        missing = [n for n in names if n not in name_to_i]
        if missing:
            raise ValueError(f"Requested ee names not in dataset: {missing}")
        return torch.tensor([name_to_i[n] for n in names], dtype=torch.long)

    def __frame_building_from_npz_arrays__(
        self,
        root: np.ndarray,
        joints: np.ndarray,
        ee_pos: np.ndarray | None,
        body_pos: np.ndarray | None,
    ):
        dt = 1.0 / self.robot_cfg.fps
        T = root.shape[0]

        frames_list = []
        for t in range(T):
            # root: [x y z qx qy qz qw]
            x, y, z, qx, qy, qz, qw = root[t].tolist()
            root_pos = torch.tensor([x, y, z], dtype=torch.float32)
            root_quat_wxyz = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
            root_quat_wxyz = quat_normalize(root_quat_wxyz)

            joints_t = torch.tensor(joints[t], dtype=torch.float32)

            # Build prev/next for velocities
            if T == 1:
                prev_pos, prev_quat, prev_j = root_pos, root_quat_wxyz, joints_t
                next_pos, next_quat, next_j = root_pos, root_quat_wxyz, joints_t
            else:
                if t == 0:
                    prev_pos, prev_quat, prev_j = root_pos, root_quat_wxyz, joints_t
                    nx = root[t + 1]
                    next_pos = torch.tensor(nx[0:3], dtype=torch.float32)
                    next_quat = quat_normalize(
                        torch.tensor([nx[6], nx[3], nx[4], nx[5]], dtype=torch.float32)
                    )
                    next_j = torch.tensor(joints[t + 1], dtype=torch.float32)
                elif t == T - 1:
                    pv = root[t - 1]
                    prev_pos = torch.tensor(pv[0:3], dtype=torch.float32)
                    prev_quat = quat_normalize(
                        torch.tensor([pv[6], pv[3], pv[4], pv[5]], dtype=torch.float32)
                    )
                    prev_j = torch.tensor(joints[t - 1], dtype=torch.float32)

                    next_pos, next_quat, next_j = root_pos, root_quat_wxyz, joints_t
                else:
                    pv = root[t - 1]
                    nx = root[t + 1]
                    prev_pos = torch.tensor(pv[0:3], dtype=torch.float32)
                    prev_quat = quat_normalize(
                        torch.tensor([pv[6], pv[3], pv[4], pv[5]], dtype=torch.float32)
                    )
                    prev_j = torch.tensor(joints[t - 1], dtype=torch.float32)

                    next_pos = torch.tensor(nx[0:3], dtype=torch.float32)
                    next_quat = quat_normalize(
                        torch.tensor([nx[6], nx[3], nx[4], nx[5]], dtype=torch.float32)
                    )
                    next_j = torch.tensor(joints[t + 1], dtype=torch.float32)

            # velocities
            root_lin_vel, root_ang_vel, joint_velocities = self.__compute_velocities__(
                prev=(prev_pos, prev_quat, prev_j, None),
                cur=(root_pos, root_quat_wxyz, joints_t, None),
                nxt=(next_pos, next_quat, next_j, None),
                dt=dt,
                row=t,
                n_rows=T,
            )

            frame = {
                "root_pos": root_pos,
                "root_quat_wxyz": root_quat_wxyz,
                "joints": joints_t,
                "root_lin_vel": root_lin_vel,
                "root_ang_vel": root_ang_vel,
                "joint_vel": joint_velocities,
            }

            # Optional EE
            if ee_pos is not None:
                # ee_pos: (T,E,3)
                frame["ee_pos"] = torch.tensor(ee_pos[t], dtype=torch.float32)

            # Optional body positions
            if body_pos is not None:
                # body_pos: (T,K,3)
                frame["body_pos"] = torch.tensor(body_pos[t], dtype=torch.float32)

            frames_list.append(frame)

        return frames_list

    def __compute_velocities__(self, prev, cur, nxt, dt, row, n_rows):
        # prev/cur/nxt are tuples (root_pos, root_quat_wxyz, joints, ee_pos_or_none)
        if self.vel_mode == "central":
            if row == 0:
                return self.__compute_velocities_forward__(
                    cur_root_pos=cur[0],
                    cur_root_quat_wxyz=cur[1],
                    cur_joints=cur[2],
                    next_root_pos=nxt[0],
                    next_root_quat_wxyz=nxt[1],
                    next_joints=nxt[2],
                    dt=dt,
                )
            elif row == n_rows - 1:
                return self.__compute_velocities_backward__(
                    prev_root_pos=prev[0],
                    prev_root_quat_wxyz=prev[1],
                    prev_joints=prev[2],
                    cur_root_pos=cur[0],
                    cur_root_quat_wxyz=cur[1],
                    cur_joints=cur[2],
                    dt=dt,
                )
            else:
                return self.__compute_velocities_central__(
                    prev_root_pos=prev[0],
                    prev_root_quat_wxyz=prev[1],
                    prev_joints=prev[2],
                    cur_root_quat_wxyz=cur[1],
                    next_root_pos=nxt[0],
                    next_root_quat_wxyz=nxt[1],
                    next_joints=nxt[2],
                    dt=dt,
                )

        # backward mode
        if row == 0:
            return self.__compute_velocities_forward__(
                cur_root_pos=cur[0],
                cur_root_quat_wxyz=cur[1],
                cur_joints=cur[2],
                next_root_pos=nxt[0],
                next_root_quat_wxyz=nxt[1],
                next_joints=nxt[2],
                dt=dt,
            )
        else:
            return self.__compute_velocities_backward__(
                prev_root_pos=prev[0],
                prev_root_quat_wxyz=prev[1],
                prev_joints=prev[2],
                cur_root_pos=cur[0],
                cur_root_quat_wxyz=cur[1],
                cur_joints=cur[2],
                dt=dt,
            )

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

        q_prev = quat_normalize(prev_root_quat_wxyz)
        q_rel = quat_mul(quat_inv(q_prev), q_cur)
        q_rel = quat_normalize(q_rel)
        w_body = quat_log(q_rel) / dt

        joint_vel = wrap_to_pi(cur_joints - prev_joints) / dt
        return v_body, w_body, joint_vel

    def _count_rows(self, file_path: Path) -> int:
        with open(file_path, "r") as f:
            return sum(1 for _ in f)

    def _load_block_csv(self, block_idx: int):
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

        full_block = self.__frame_building_from_csv_rows__(data)

        core_offset = core_start - load_start
        core_len = core_end - core_start
        self.dataset = full_block[core_offset : core_offset + core_len]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        if self.input_format == "npz":
            # NPZ always full-loaded in this implementation
            return self.dataset[idx]

        # CSV
        if not self.lazy_load:
            return self.dataset[idx]

        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of range (0, {self.num_frames - 1})")

        if not (self.current_block_start <= idx < self.current_block_end):
            block_idx = idx // self.lazy_load_window
            self._load_block_csv(block_idx)

        local_idx = int(idx - self.current_block_start)
        return self.dataset[local_idx]
