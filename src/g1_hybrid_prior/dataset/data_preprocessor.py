import os
import pinocchio as pin
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

from .robot_cfg import RobotCfg, load_robot_cfg
from ..helpers import get_project_root


# Usage examples:
# python -m g1_hybrid_prior.data.data_preprocessor --data /path/to/g1
# python -m g1_hybrid_prior.data.data_preprocessor --data /path/to/g1 --export_npz_bodies
# python -m g1_hybrid_prior.data.data_preprocessor --data /path/to/g1 --export_csv_ee --export_npz_bodies
# python -m g1_hybrid_prior.data.data_preprocessor --data /path/to/g1 --export_npz_bodies --body_links pelvis torso_link left_ankle_roll_link right_ankle_roll_link


def _frame_is_body(f: pin.Frame) -> bool:
    # Pinocchio enum: pin.FrameType.BODY
    try:
        return f.type == pin.FrameType.BODY
    except Exception:
        # fallback (older pinocchio)
        return str(getattr(f, "type", "")).lower().endswith("body")


class FKAugmenter:
    """
    Backward compatible:
      - può fare EE->CSV (come prima)
      - può fare BODY->NPZ (nuovo)
      - può fare entrambi
      - NEW: può includere anche EE dentro NPZ (anche se non esporti CSV)

    Il parsing del CSV resta basato su robots.yaml (root_fields + joint_order).
    L'URDF serve solo per la kinematic tree e per i nomi di joint/link.
    """

    def __init__(
        self,
        urdf_path: str,
        robot_cfg: RobotCfg,
        ee_link_names: Optional[List[str]] = None,
        body_link_names: Optional[List[str]] = None,  # se None: tutti i BODY frames
        export_body_quat: bool = False,
    ):
        self.urdf_path = urdf_path
        self.robot_cfg = robot_cfg
        self.ee_link_names = ee_link_names or []
        self.body_link_names = body_link_names  # None = auto
        self.export_body_quat = export_body_quat

        # Pinocchio model (floating base)
        self.model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        print(f"[FK] Model loaded from {Path(urdf_path).name}")
        print(f"[FK] Pinocchio dims -> nq: {self.model.nq}, nv: {self.model.nv}")

        # Build joint name -> idx_q mapping for *dataset* joints (order from yaml)
        self.csv_to_pin_map: List[Tuple[int, int]] = []
        dataset_joints = self.robot_cfg.joint_order

        for csv_idx, joint_name in enumerate(dataset_joints):
            if not self.model.existJointName(joint_name):
                raise ValueError(
                    f"Joint '{joint_name}' presente nel dataset ma NON nell'URDF!"
                )

            joint_id = self.model.getJointId(joint_name)
            pin_q_idx = self.model.joints[joint_id].idx_q  # start index in q
            self.csv_to_pin_map.append((csv_idx, pin_q_idx))

        print(f"[FK] Joint Mapping creato per {len(self.csv_to_pin_map)} giunti.")

        # EE frames (optional, but we may want them for NPZ too)
        self.ee_frame_ids: List[int] = []
        if len(self.ee_link_names) > 0:
            missing = []
            for name in self.ee_link_names:
                if self.model.existFrame(name):
                    self.ee_frame_ids.append(self.model.getFrameId(name))
                else:
                    missing.append(name)
            if missing:
                available = [f.name for f in self.model.frames]
                raise ValueError(
                    f"Link/Frame EE non trovati nell'URDF: {missing}\n"
                    f"Esempio frames disponibili (first 80): {available[:80]}"
                )
            print(
                f"[FK] EE targets: {self.ee_link_names} (count={len(self.ee_link_names)})"
            )

        # BODY frames (optional)
        # Se body_link_names è None -> prendi tutti i body frames
        self.body_frame_ids: List[int] = []
        self.body_names: List[str] = []

        if self.body_link_names is None:
            # Auto: all BODY frames
            for i, fr in enumerate(self.model.frames):
                if _frame_is_body(fr):
                    self.body_names.append(fr.name)
                    self.body_frame_ids.append(i)
        else:
            # User-defined list
            missing = []
            for name in self.body_link_names:
                if self.model.existFrame(name):
                    fid = self.model.getFrameId(name)
                    self.body_names.append(name)
                    self.body_frame_ids.append(fid)
                else:
                    missing.append(name)
            if missing:
                available = [f.name for f in self.model.frames]
                raise ValueError(
                    f"Link/Frame BODY non trovati nell'URDF: {missing}\n"
                    f"Esempio frames disponibili (first 80): {available[:80]}"
                )

        if len(self.body_frame_ids) > 0:
            print(f"[FK] BODY frames selected: {len(self.body_frame_ids)}")

    def process_path(
        self,
        input_path: str,
        export_csv_ee: bool = True,
        export_npz_bodies: bool = False,
        output_dir_csv: Optional[Path] = None,
        output_dir_npz: Optional[Path] = None,
    ):
        path = Path(input_path)
        if path.is_file():
            self._process_single_file(
                path,
                export_csv_ee=export_csv_ee,
                export_npz_bodies=export_npz_bodies,
                output_dir_csv=output_dir_csv,
                output_dir_npz=output_dir_npz,
            )
        elif path.is_dir():
            print(f"[FK] Processing directory: {path}")

            # default output dirs (parallel)
            if output_dir_csv is None:
                output_dir_csv = path.parent / (path.name + "_ee_full_augmented_2")
            if output_dir_npz is None:
                output_dir_npz = path.parent / (path.name + "_body_fk_npz")

            if export_csv_ee:
                os.makedirs(output_dir_csv, exist_ok=True)
                print(f"[FK] Output CSV directory: {output_dir_csv}")
            if export_npz_bodies:
                os.makedirs(output_dir_npz, exist_ok=True)
                print(f"[FK] Output NPZ directory: {output_dir_npz}")

            files = list(path.glob("*.csv"))
            for f in tqdm(files, desc="Total Files", unit="file"):
                self._process_single_file(
                    f,
                    export_csv_ee=export_csv_ee,
                    export_npz_bodies=export_npz_bodies,
                    output_dir_csv=output_dir_csv,
                    output_dir_npz=output_dir_npz,
                )
        else:
            raise ValueError(f"Path invalido: {input_path}")

    def _process_single_file(
        self,
        file_path: Path,
        export_csv_ee: bool,
        export_npz_bodies: bool,
        output_dir_csv: Optional[Path],
        output_dir_npz: Optional[Path],
    ):
        # 1) read raw CSV (no header)
        try:
            df = pd.read_csv(file_path, header=None)
        except Exception as e:
            print(f"Errore lettura {file_path}: {e}")
            return

        expected_cols = self.robot_cfg.root_fields + self.robot_cfg.joint_order

        if len(df.columns) == len(expected_cols):
            df.columns = expected_cols
        elif len(df.columns) > len(expected_cols):
            df.columns = expected_cols + [
                f"extra_{i}" for i in range(len(df.columns) - len(expected_cols))
            ]
        else:
            print(
                f"⚠️ Skipping {file_path.name}: Expected {len(expected_cols)} cols, found {len(df.columns)}"
            )
            return

        root_cols = self.robot_cfg.root_fields
        joint_cols = self.robot_cfg.joint_order

        root_data = df[root_cols].to_numpy()  # (N,7)
        joint_data = df[joint_cols].to_numpy()  # (N,dof)
        n_frames = len(df)

        # Prepare outputs
        ee_positions: Dict[str, np.ndarray] = {}
        if export_csv_ee:
            if len(self.ee_link_names) == 0:
                raise RuntimeError("export_csv_ee=True ma ee_link_names è vuoto.")
            ee_positions = {
                name: np.zeros((n_frames, 3), dtype=np.float32)
                for name in self.ee_link_names
            }

        body_pos = None
        body_quat = None
        if export_npz_bodies:
            if len(self.body_frame_ids) == 0:
                raise RuntimeError("export_npz_bodies=True ma non ho body_frame_ids.")
            K = len(self.body_frame_ids)
            body_pos = np.zeros((n_frames, K, 3), dtype=np.float32)
            if self.export_body_quat:
                body_quat = np.zeros((n_frames, K, 4), dtype=np.float32)

        # --- NEW: EE positions saved inside NPZ too (even if export_csv_ee=False) ---
        ee_pos = None
        if export_npz_bodies and len(self.ee_frame_ids) > 0:
            E = len(self.ee_frame_ids)
            ee_pos = np.zeros((n_frames, E, 3), dtype=np.float32)

        # Pinocchio q
        q = np.zeros(self.model.nq, dtype=np.float64)

        # 2) FK loop
        for i in tqdm(
            range(n_frames),
            desc=f"Processing {file_path.name}",
            leave=False,
            unit="frame",
        ):
            q[:7] = root_data[i]  # assumes [x y z qx qy qz qw] already

            row_joints = joint_data[i]
            for csv_j_idx, pin_q_idx in self.csv_to_pin_map:
                q[pin_q_idx] = row_joints[csv_j_idx]

            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            if export_csv_ee:
                for ee_idx, name in enumerate(self.ee_link_names):
                    fid = self.ee_frame_ids[ee_idx]
                    ee_positions[name][i] = self.data.oMf[fid].translation

            if export_npz_bodies:
                # all selected body frames
                for k, fid in enumerate(self.body_frame_ids):
                    M = self.data.oMf[fid]
                    body_pos[i, k, :] = M.translation
                    if self.export_body_quat:
                        # rotation -> quaternion (x,y,z,w)
                        quat = pin.Quaternion(M.rotation)
                        body_quat[i, k, :] = np.array(
                            [quat.x, quat.y, quat.z, quat.w], dtype=np.float32
                        )

                # --- NEW: also store EE positions in NPZ ---
                if ee_pos is not None:
                    for e, fid in enumerate(self.ee_frame_ids):
                        ee_pos[i, e, :] = self.data.oMf[fid].translation

        # 3) Save CSV (EE)
        if export_csv_ee:
            for name in self.ee_link_names:
                df[f"{name}_pos_x"] = ee_positions[name][:, 0]
                df[f"{name}_pos_y"] = ee_positions[name][:, 1]
                df[f"{name}_pos_z"] = ee_positions[name][:, 2]

            if output_dir_csv is not None:
                save_csv = output_dir_csv / file_path.name
            else:
                save_csv = file_path.parent / (
                    file_path.stem + "_end_eff_full_augmented.csv"
                )

            df.to_csv(save_csv, index=False, header=False)

        # 4) Save NPZ (Bodies)
        if export_npz_bodies:
            if output_dir_npz is not None:
                save_npz = output_dir_npz / (file_path.stem + ".npz")
            else:
                save_npz = file_path.parent / (file_path.stem + "_body_fk.npz")

            payload = {
                "root": root_data.astype(np.float32),  # (N,7)
                "q_joints_yaml": joint_data.astype(
                    np.float32
                ),  # (N,dof) in robots.yaml order
                "body_names": np.array(self.body_names),
                "body_pos": body_pos,
            }
            if self.export_body_quat and body_quat is not None:
                payload["body_quat"] = body_quat

            # --- NEW: include EE in NPZ payload if available ---
            if ee_pos is not None:
                payload["ee_names"] = np.array(self.ee_link_names)
                payload["ee_pos"] = ee_pos

            np.savez_compressed(save_npz, **payload)


def get_args():
    parser = argparse.ArgumentParser(description="G1 FK augmenter (EE->CSV, BODY->NPZ)")

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

    # EE export (old flow)
    parser.add_argument(
        "--export_csv_ee",
        action="store_true",
        help="Export CSV with appended EE positions",
    )
    parser.add_argument(
        "--ees",
        nargs="*",
        default=None,
        help="Override EE link names (else ee_link_names in yaml)",
    )

    # BODY export (new flow)
    parser.add_argument(
        "--export_npz_bodies",
        action="store_true",
        help="Export NPZ with body positions for all links",
    )
    parser.add_argument(
        "--body_links",
        nargs="*",
        default=None,
        help="Optional explicit body link names. If omitted: all Pinocchio BODY frames.",
    )
    parser.add_argument(
        "--export_body_quat",
        action="store_true",
        help="Also store body quaternion in NPZ",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    print(f"--- Start Data Processing: {args.robot_name} ---")
    print(f"Dataset: {args.data}")

    try:
        cfg = load_robot_cfg(args.yaml, args.robot_name)

        # Default behavior: if user didn't specify any export flag, keep old flow
        export_csv_ee = args.export_csv_ee
        export_npz_bodies = args.export_npz_bodies
        if (not export_csv_ee) and (not export_npz_bodies):
            export_csv_ee = True  # backward compat

        ee_names = (
            args.ees
            if (args.ees is not None and len(args.ees) > 0)
            else cfg.ee_link_names
        )

        # IMPORTANT CHANGE:
        # Pass ee_link_names always, so NPZ can include EE even if export_csv_ee=False.
        augmenter = FKAugmenter(
            urdf_path=args.urdf,
            robot_cfg=cfg,
            ee_link_names=ee_names,
            body_link_names=args.body_links,  # None => all BODY frames
            export_body_quat=args.export_body_quat,
        )

        augmenter.process_path(
            args.data,
            export_csv_ee=export_csv_ee,
            export_npz_bodies=export_npz_bodies,
        )

        print("\n--- Processing Completed Successfully ---")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
