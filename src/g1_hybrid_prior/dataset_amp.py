import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List

from .helpers import quat_rotate_inv, quat_normalize, get_project_root
from .robot_cfg import load_robot_cfg


class G1AMPDataset(Dataset):
    def __init__(
        self,
        file_path,
        num_amp_obs_steps: int = 2,
        device="cuda",
        robot="g1",
        Z_UP=True,
        strict=True,
    ):
        self.device = device
        self.file_path = file_path

        print(f"[G1AMPDataset] Loading {file_path}...")
        data = np.load(file_path, allow_pickle=True)

        # ---- Load names from NPZ ----
        npz_dof_names: List[str] = [str(x) for x in data["dof_names"].tolist()]
        self.npz_dof_names = npz_dof_names

        npz_body_names: List[str] = [str(x) for x in data["body_names"].tolist()]
        self.npz_body_names = npz_body_names

        # ---- Canonical order from robots.yaml ----
        robots_yaml = str(get_project_root() / "config" / "robots.yaml")
        self.robot_cfg = load_robot_cfg(robots_yaml, robot)
        canonical = list(self.robot_cfg.joint_order)

        # ---- Build permutation: NPZ -> canonical ----
        name_to_idx = {n: i for i, n in enumerate(npz_dof_names)}
        missing = [n for n in canonical if n not in name_to_idx]
        extra = [n for n in npz_dof_names if n not in set(canonical)]

        self.num_amp_obs_steps = num_amp_obs_steps
        assert (
            self.num_amp_obs_steps >= 2
        ), "[G1AMPDataset] AMP observation window must be at least 2 steps."

        if strict:
            if missing:
                raise ValueError(
                    f"[G1AMPDataset] Missing DOF(s) in NPZ: {missing[:10]} ..."
                )
        else:
            if missing:
                print(
                    f"[G1AMPDataset] WARNING: Missing DOF(s) in NPZ: {missing[:10]} ..."
                )

        perm = [name_to_idx[n] for n in canonical if n in name_to_idx]
        self._perm_npz_to_canonical = torch.tensor(
            perm, dtype=torch.long, device=device
        )

        # ---- Load tensors ----
        # Root: index 0 body
        self.root_pos_w = torch.tensor(
            data["body_positions"][:, 0, :], dtype=torch.float32, device=device
        )
        self.root_rot_w = torch.tensor(
            data["body_rotations"][:, 0, :], dtype=torch.float32, device=device
        )
        self.root_lin_vel_w = torch.tensor(
            data["body_linear_velocities"][:, 0, :], dtype=torch.float32, device=device
        )
        self.root_ang_vel_w = torch.tensor(
            data["body_angular_velocities"][:, 0, :], dtype=torch.float32, device=device
        )
        # Full rigid-body positions (WORLD) for AMP early termination:
        # Shape: (T, B, 3)
        self.body_pos_w = torch.tensor(
            data["body_positions"], dtype=torch.float32, device=device
        )

        dof_pos_npz = torch.tensor(
            data["dof_positions"], dtype=torch.float32, device=device
        )
        dof_vel_npz = torch.tensor(
            data["dof_velocities"], dtype=torch.float32, device=device
        )

        # ---- Reorder DOF to canonical ----
        self.dof_pos = dof_pos_npz.index_select(1, self._perm_npz_to_canonical)
        self.dof_vel = dof_vel_npz.index_select(1, self._perm_npz_to_canonical)

        # ---- Pre-processing for AMP obs (body frame) ----
        root_rot_norm = quat_normalize(self.root_rot_w)
        lin_vel_body = quat_rotate_inv(root_rot_norm, self.root_lin_vel_w)
        ang_vel_body = quat_rotate_inv(root_rot_norm, self.root_ang_vel_w)

        root_h = self.root_pos_w[:, 2:3] if Z_UP else self.root_pos_w[:, 1:2]

        # AMP batch in canonical DOF order
        self.amp_batch = torch.cat(
            [
                root_h,
                root_rot_norm,
                lin_vel_body,
                ang_vel_body,
                self.dof_pos,
                self.dof_vel,
            ],
            dim=-1,
        )

        self.num_frames = self.amp_batch.shape[0]
        print(
            f"[G1AMPDataset] Loaded {self.num_frames} frames. DOFs={self.dof_pos.shape[1]}"
        )
        print(f"[G1AMPDataset] Canonical DOF[0:5]: {canonical[:5]}")
        print(f"[G1AMPDataset] NPZ DOF[0:5]: {npz_dof_names[:5]}")
        print(f"[G1AMPDataset] NPZ BODY[0:5]: {npz_body_names[:5]}")

    def __len__(self):
        return self.amp_batch.shape[0]

    def __getitem__(self, idx):
        return self.amp_batch[idx]

    def sample(self, batch_size):
        T = self.amp_batch.shape[0]
        K = self.num_amp_obs_steps

        # scegli t in [K-1, T-1] così hai storia completa
        t = torch.randint(K - 1, T, (batch_size,), device=self.device)

        # costruisci indici [t, t-1, ..., t-K+1]
        offsets = torch.arange(0, K, device=self.device)  # [0..K-1]
        idx = t.unsqueeze(1) - offsets.unsqueeze(0)  # (B, K)

        # prendi frames e flattna: (B, K, 69) -> (B, K*69)
        batch = self.amp_batch.index_select(0, idx.reshape(-1)).reshape(
            batch_size, K, -1
        )
        return batch.reshape(batch_size, -1)


# if __name__ == "__main__":
#     ds = G1AMPDataset(
#         "/home/valerio/g1_hybrid_prior/data_amp/LAFAN-G1/LAFAN_dance1_subject1_0_-1.npz",
#         device="cuda",
#     )
#     print(ds.body_pos_w.shape)  # (T, B, 3)
#     print(len(ds.npz_body_names))  # B
#     print(ds.root_pos_w.shape)  # (T, 3)
#     print(ds.root_rot_w.shape)  # (T, 4)
