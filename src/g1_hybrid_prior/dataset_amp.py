import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List
from pathlib import Path  # Aggiunto per gestire i path

from .helpers import quat_rotate_inv, quat_normalize, get_project_root
from .robot_cfg import load_robot_cfg


class G1AMPDataset(Dataset):
    def __init__(
        self,
        file_path,
        num_amp_obs_steps: int = 2,
        device="cuda",
        robot="g1_amp",
        Z_UP=True,
        strict=True,
    ):
        self.device = device
        self.file_path = Path(file_path)  # Convertiamo in Path object

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"[G1AMPDataset] File or Directory not found: {self.file_path}"
            )

        if robot != "g1_amp":
            raise ValueError(
                f"[G1AMPDataset] Unsupported robot: {robot}. Only 'g1_amp' is supported with an AMP dataset."
            )

        # 1. RACCOLTA LISTA FILE
        file_list = []
        if self.file_path.is_dir():
            # Glob di tutti i .npz ordinati
            file_list = sorted(list(self.file_path.glob("*.npz")))
            if len(file_list) == 0:
                raise FileNotFoundError(
                    f"[G1AMPDataset] No .npz files found in directory: {self.file_path}"
                )
            print(f"[G1AMPDataset] Found directory with {len(file_list)} clips.")
        else:
            if self.file_path.suffix != ".npz":
                raise ValueError(
                    f"[G1AMPDataset] Invalid file type: {self.file_path}. Must be .npz"
                )
            file_list = [self.file_path]

        # 2. CARICAMENTO E CONCATENAZIONE
        # Liste temporanee per accumulare i dati di tutti i file
        all_body_pos = []
        all_body_rot = []
        all_body_vel = []
        all_body_ang_vel = []
        all_dof_pos = []
        all_dof_vel = []

        # Variabili metadata (prese dal primo file)
        self.npz_dof_names = None
        self.npz_body_names = None

        print(f"[G1AMPDataset] Loading {len(file_list)} files...")

        for i, f in enumerate(file_list):
            try:
                data = np.load(f, allow_pickle=True)

                # Al primo file, settiamo i metadati e facciamo i check
                if i == 0:
                    self.npz_dof_names = [str(x) for x in data["dof_names"].tolist()]
                    self.npz_body_names = [str(x) for x in data["body_names"].tolist()]

                    # ---- Setup Permutazione (Fatto una sola volta basandosi sul primo file) ----
                    robots_yaml = str(get_project_root() / "config" / "robots.yaml")
                    self.robot_cfg = load_robot_cfg(robots_yaml, robot)
                    canonical = list(self.robot_cfg.joint_order)

                    name_to_idx = {n: i for i, n in enumerate(self.npz_dof_names)}
                    missing = [n for n in canonical if n not in name_to_idx]

                    if strict and missing:
                        raise ValueError(
                            f"[G1AMPDataset] Missing DOF(s) in NPZ: {missing[:10]} ..."
                        )
                    elif missing:
                        print(
                            f"[G1AMPDataset] WARNING: Missing DOF(s) in NPZ: {missing[:10]} ..."
                        )

                    perm = [name_to_idx[n] for n in canonical if n in name_to_idx]
                    self._perm_npz_to_canonical = torch.tensor(
                        perm, dtype=torch.long, device=device
                    )
                else:
                    # Check consistenza (opzionale ma consigliato): verifichiamo che gli altri file abbiano gli stessi giunti
                    curr_dof_names = [str(x) for x in data["dof_names"].tolist()]
                    if curr_dof_names != self.npz_dof_names:
                        print(
                            f"[G1AMPDataset] ⚠️ WARNING: Mismatch in DOF names in file {f.name}. Skipping."
                        )
                        continue
                    curr_body_names = [str(x) for x in data["body_names"].tolist()]
                    if curr_body_names != self.npz_body_names:
                        print(
                            f"[G1AMPDataset] ⚠️ WARNING: Mismatch in body names in file {f.name}. Skipping."
                        )
                        continue

                # Appendiamo i dati
                all_body_pos.append(data["body_positions"])
                all_body_rot.append(data["body_rotations"])
                all_body_vel.append(data["body_linear_velocities"])
                all_body_ang_vel.append(data["body_angular_velocities"])
                all_dof_pos.append(data["dof_positions"])
                all_dof_vel.append(data["dof_velocities"])

            except Exception as e:
                print(f"[G1AMPDataset] Error loading {f}: {e}")

        # Concateniamo tutto lungo l'asse temporale (axis=0)
        # Usiamo np.concatenate prima di convertire in Tensor per efficienza
        raw_body_pos = np.concatenate(all_body_pos, axis=0)
        raw_body_rot = np.concatenate(all_body_rot, axis=0)
        raw_body_vel = np.concatenate(all_body_vel, axis=0)
        raw_body_ang = np.concatenate(all_body_ang_vel, axis=0)
        raw_dof_pos = np.concatenate(all_dof_pos, axis=0)
        raw_dof_vel = np.concatenate(all_dof_vel, axis=0)

        # 3. CREAZIONE TENSORI (Come prima, ma sui dati concatenati)
        self.num_amp_obs_steps = num_amp_obs_steps
        assert (
            self.num_amp_obs_steps >= 2
        ), "[G1AMPDataset] AMP observation window must be at least 2 steps."

        # Root: index 0 body
        self.root_pos_w = torch.tensor(
            raw_body_pos[:, 0, :], dtype=torch.float32, device=device
        )
        self.root_rot_w = torch.tensor(
            raw_body_rot[:, 0, :], dtype=torch.float32, device=device
        )
        self.root_lin_vel_w = torch.tensor(
            raw_body_vel[:, 0, :], dtype=torch.float32, device=device
        )
        self.root_ang_vel_w = torch.tensor(
            raw_body_ang[:, 0, :], dtype=torch.float32, device=device
        )

        # Full rigid-body positions
        self.body_pos_w = torch.tensor(raw_body_pos, dtype=torch.float32, device=device)
        self.body_rot_w = torch.tensor(raw_body_rot, dtype=torch.float32, device=device)

        dof_pos_npz = torch.tensor(raw_dof_pos, dtype=torch.float32, device=device)
        dof_vel_npz = torch.tensor(raw_dof_vel, dtype=torch.float32, device=device)

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
            f"[G1AMPDataset] Successfully loaded {self.num_frames} frames from {len(file_list)} files."
        )
        print(f"[G1AMPDataset] DOFs={self.dof_pos.shape[1]}")

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
