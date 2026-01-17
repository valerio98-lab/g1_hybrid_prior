import argparse
import numpy as np
from .robot_cfg import load_robot_cfg, RobotCfg
from ..helpers import get_project_root


def split_row(row: np.ndarray, robot_cfg: RobotCfg):
    root = row[: robot_cfg.root_dim]
    joints = row[robot_cfg.root_dim : robot_cfg.dof + robot_cfg.root_dim]

    qx, qy, qz, qw = root[3:7]
    root_pos = root[0:3]
    root_quat_xyzw = np.array([qx, qy, qz, qw], dtype=row.dtype)  # x,y,z,w
    root_quat_wxyz = np.array([qw, qx, qy, qz], dtype=row.dtype)  # w,x,y,z

    return root_pos, root_quat_xyzw, root_quat_wxyz, joints


def inspect_robot_cfg(
    robot_cfg: RobotCfg,
    path: str,
    n_frames: int = 1,
):

    data = np.loadtxt(path, delimiter=",")

    if data.shape[1] != robot_cfg.expected_cols:
        raise ValueError(
            f"CSV has {data.shape[1]} cols, but {robot_cfg.name} expects "
            f"{robot_cfg.expected_cols} (root {robot_cfg.root_dim} + dof {robot_cfg.dof})."
        )

    print(
        f"{robot_cfg.name}: fps={robot_cfg.fps}, dof={robot_cfg.dof}, total_cols={robot_cfg.expected_cols}"
    )
    n_frames = min(n_frames, data.shape[0])
    for i in range(n_frames):
        row = data[i]
        root_pos, root_quat_xyzw, root_quat_wxyz, joints = split_row(row, robot_cfg)
        print(f"\nFrame {i}:")
        print(f"  Root Position: {root_pos}")
        print(f"  Root Quaternion (xyzw): {root_quat_xyzw}")
        print(f"  Root Quaternion (wxyz): {root_quat_wxyz}")
        print("  first 6 joints (name: val):")
        for name, val in list(zip(robot_cfg.joint_order, joints))[:6]:
            print(f"    {name}: {val:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to a G1 CSV file (e.g. dance1_subject2_g1.csv)",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        required=True,
        help="Type of robot (possible values: 'g1', 'h1', h1_2)",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=1,
        help="How many frames to print",
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        default=36,
        help="How many columns of each frame to print",
    )

    args = parser.parse_args()
    file_yaml = str(get_project_root() / "config" / "robots.yaml")
    robot_cfg = load_robot_cfg(file_yaml, args.robot_type)

    inspect_robot_cfg(robot_cfg, args.file, n_frames=args.n_frames)
