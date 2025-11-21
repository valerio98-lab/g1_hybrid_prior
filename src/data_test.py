import argparse
import numpy as np
from robot_cfg import load_robot_cfg


def inspect_g1_file(path: str, n_frames: int = 1, n_cols: int = 10):

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    print(f"Data shape: {data.shape}")  # (num_frames, num_columns)

    n_frames = min(n_frames, data.shape[0])
    for i in range(n_frames):
        row = data[i]
        print(f"\nFrame {i}:")
        print(f"  First {min(n_cols, len(row))} values:")
        print("  ", row[:n_cols])

    print("\nDone.\n")


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
    file_yaml = ...
    robot_cfg = load_robot_cfg(file_yaml, args.robot_type)
    inspect_g1_file(args.file, n_frames=args.n_frames, n_cols=args.n_cols)
