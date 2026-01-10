import os
import pinocchio as pin
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from .robot_cfg import RobotCfg, load_robot_cfg
from .helpers import get_project_root


class EndEffectorAugmenter:
    def __init__(self, urdf_path: str, ee_link_names: List[str], robot_cfg: RobotCfg):
        """
        Calcola la Forward Kinematics (FK) per aggiungere le posizioni degli EE al dataset.
        """
        self.urdf_path = urdf_path
        self.ee_link_names = ee_link_names
        self.robot_cfg = robot_cfg

        # 1. Carica il modello Pinocchio (Floating Base)
        self.model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        print(f"[FK] Model loaded from {Path(urdf_path).name}")
        print(f"[FK] Pinocchio dims -> nq: {self.model.nq}, nv: {self.model.nv}")

        # 2. Trova gli ID dei frame degli end effector
        self.ee_frame_ids = []
        missing_links = []
        for name in ee_link_names:
            if self.model.existFrame(name):
                self.ee_frame_ids.append(self.model.getFrameId(name))
            else:
                missing_links.append(name)

        if missing_links:
            available = [f.name for f in self.model.frames]
            raise ValueError(
                f"Link/Frame non trovati nell'URDF: {missing_links}\n"
                f"Esempio frames disponibili (first 80): {available[:80]}"
            )

        # 3. Costruisci la mappa {indice_colonna_csv: indice_q_pinocchio}
        self.csv_to_pin_map = []
        dataset_joints = self.robot_cfg.joint_order

        for csv_idx, joint_name in enumerate(dataset_joints):
            if not self.model.existJointName(joint_name):
                raise ValueError(
                    f"Joint '{joint_name}' presente nel dataset ma NON nell'URDF!"
                )

            joint_id = self.model.getJointId(joint_name)
            pin_q_idx = self.model.joints[joint_id].idx_q
            self.csv_to_pin_map.append((csv_idx, pin_q_idx))

        print(f"[FK] Joint Mapping creato per {len(self.csv_to_pin_map)} giunti.")

    def process_path(self, input_path: str):
        """Processa un singolo file CSV o una cartella intera."""
        path = Path(input_path)
        if path.is_file():
            self._process_single_file(path)
        elif path.is_dir():
            print(f"[FK] Processing directory: {path}")
            # Crea output dir parallela con suffisso
            output_dir = path.parent / (path.name + "_ee_full_augmented")
            os.makedirs(output_dir, exist_ok=True)
            print(f"[FK] Output directory: {output_dir}")

            files = list(path.glob("*.csv"))
            for f in tqdm(files, desc="Total Files", unit="file"):
                self._process_single_file(f, output_dir)
        else:
            raise ValueError(f"Path invalido: {input_path}")

    def _process_single_file(self, file_path: Path, output_dir: Optional[Path] = None):
        # 1. Carica CSV (SENZA HEADER)
        try:
            # header=None è cruciale: i file sono solo numeri
            df = pd.read_csv(file_path, header=None)
        except Exception as e:
            print(f"Errore lettura {file_path}: {e}")
            return

        # 2. Assegna i nomi alle colonne manualmente basandoci su RobotCfg
        # Questo ci serve SOLO internamente per poter accedere ai dati,
        # ma NON li scriveremo nel file finale.
        expected_cols = self.robot_cfg.root_fields + self.robot_cfg.joint_order

        if len(df.columns) == len(expected_cols):
            df.columns = expected_cols
        elif len(df.columns) > len(expected_cols):
            # Gestione colonne extra già presenti (robustezza)
            df.columns = expected_cols + [
                f"extra_{i}" for i in range(len(df.columns) - len(expected_cols))
            ]
        else:
            print(
                f"⚠️ Skipping {file_path.name}: Expected {len(expected_cols)} cols, found {len(df.columns)}"
            )
            return

        # 3. Estrai dati (NumPy per velocità)
        root_cols = self.robot_cfg.root_fields
        root_data = df[root_cols].to_numpy()  # (N, 7)
        joint_cols = self.robot_cfg.joint_order
        joint_data = df[joint_cols].to_numpy()  # (N, dof)

        n_frames = len(df)
        ee_positions = {name: np.zeros((n_frames, 3)) for name in self.ee_link_names}
        q = np.zeros(self.model.nq)

        # 4. Loop cinematica
        for i in tqdm(
            range(n_frames),
            desc=f"Processing {file_path.name}",
            leave=False,
            unit="frame",
        ):
            q[:7] = root_data[i]

            row_joints = joint_data[i]
            for csv_j_idx, pin_q_idx in self.csv_to_pin_map:
                q[pin_q_idx] = row_joints[csv_j_idx]

            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            for ee_idx, name in enumerate(self.ee_link_names):
                frame_id = self.ee_frame_ids[ee_idx]
                pos = self.data.oMf[frame_id].translation
                ee_positions[name][i] = pos

        # 5. Aggiungi colonne al DataFrame
        # L'ordine sarà: [Dati Originali] + [EE1_x, EE1_y, EE1_z] + [EE2_x...]
        for name in self.ee_link_names:
            df[f"{name}_pos_x"] = ee_positions[name][:, 0]
            df[f"{name}_pos_y"] = ee_positions[name][:, 1]
            df[f"{name}_pos_z"] = ee_positions[name][:, 2]

        # 6. Salvataggio
        if output_dir:
            save_path = output_dir / file_path.name
        else:
            save_path = file_path.parent / (
                file_path.stem + "_end_eff_full_augmented.csv"
            )

        # --- FIX: header=False ---
        # Manteniamo il formato "raw numbers" originale.
        df.to_csv(save_path, index=False, header=False)


def get_args():
    parser = argparse.ArgumentParser(
        description="G1 End-Effector Augmenter via Pinocchio FK"
    )

    DEFAULT_URDF = str(
        get_project_root() / "assets" / "g1_29dof_with_hand_rev_1_0.urdf"
    )
    DEFAULT_YAML = str(get_project_root() / "config" / "robots.yaml")
    DEFAULT_DATA = str(
        get_project_root() / "data_raw" / "LAFAN1_Retargeting_Dataset" / "g1"
    )

    parser.add_argument(
        "--urdf", type=str, default=DEFAULT_URDF, help="Path to robot URDF"
    )
    parser.add_argument(
        "--yaml", type=str, default=DEFAULT_YAML, help="Path to robots.yaml config"
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA, help="Path to csv file or directory"
    )
    parser.add_argument(
        "--robot_name", type=str, default="g1", help="Robot name in yaml config"
    )

    parser.add_argument(
        "--ees",
        nargs="*",
        default=None,
        help="Optional override EE link names (otherwise uses ee_link_names from yaml)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    print(f"--- Start Data Processing: {args.robot_name} ---")
    print(f"Dataset: {args.data}")

    try:
        # Carica Config
        cfg = load_robot_cfg(args.yaml, args.robot_name)

        ee_names = (
            args.ees
            if (args.ees is not None and len(args.ees) > 0)
            else cfg.ee_link_names
        )
        print(f"Targets (EE links): {ee_names}  (count={len(ee_names)})")

        # Inizializza Processore
        augmenter = EndEffectorAugmenter(args.urdf, ee_names, cfg)

        # Lancia Processo
        augmenter.process_path(args.data)

        print("\n--- Processing Completed Successfully ---")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
