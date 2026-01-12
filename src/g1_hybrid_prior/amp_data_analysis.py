import numpy as np
import torch

from g1_hybrid_prior.helpers import quat_rotate_inv

FILE_PATH = (
    "/home/valerio/g1_hybrid_prior/data_amp/LAFAN-G1/LAFAN_walk1_subject1_0_-1.npz"
)


def _to_str_list(arr) -> list[str]:
    """Convert np array of strings/bytes/objects into a clean python list[str]."""
    if arr is None:
        return []
    a = np.asarray(arr)

    # handle scalar string
    if a.shape == ():
        v = a.item()
        if isinstance(v, bytes):
            return [v.decode("utf-8", errors="replace")]
        return [str(v)]

    out = []
    for x in a.tolist():
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def inspect_dataset():
    try:
        print(f"--- Inspecting: {FILE_PATH} ---")
        data = np.load(FILE_PATH, allow_pickle=True)

        keys = list(data.files)
        print(f"\n[KEYS FOUND]: {keys}")

        # --- Print joint/body names if present ---
        dof_names = _to_str_list(data["dof_names"]) if "dof_names" in keys else []
        body_names = _to_str_list(data["body_names"]) if "body_names" in keys else []

        if dof_names:
            print(f"\n[DOF / JOINT NAMES] (count={len(dof_names)})")
            for i, n in enumerate(dof_names):
                print(f"  [{i:02d}] {n}")
        else:
            print("\n[DOF / JOINT NAMES] not found (key 'dof_names' missing)")

        if body_names:
            print(f"\n[BODY NAMES] (count={len(body_names)})")
            for i, n in enumerate(body_names):
                print(f"  [{i:02d}] {n}")
        else:
            print("\n[BODY NAMES] not found (key 'body_names' missing)")

        # --- Print structure for each key ---
        for k in keys:
            obj = data[k]
            if isinstance(obj, np.ndarray):
                print(f"\nKey: '{k}'")
                print(f"  Shape: {obj.shape}")
                print(f"  Type:  {obj.dtype}")

                if np.issubdtype(obj.dtype, np.number):
                    print(
                        f"  Min: {obj.min():.4f} | Max: {obj.max():.4f} "
                        f"| Mean: {obj.mean():.4f}"
                    )

                    # Check quaternions
                    if "quat" in k or (obj.ndim > 0 and obj.shape[-1] == 4):
                        norms = np.linalg.norm(obj, axis=-1)
                        print(
                            f"  -> Seems like a quaternion? "
                            f"Norm mean: {norms.mean():.4f}"
                        )

            else:
                print(f"\nKey: '{k}' -> Non-array type: {type(obj)}")

        # --- Quick consistency checks (optional but handy) ---
        print("\n--- CONSISTENCY CHECKS ---")
        if dof_names and "dof_positions" in keys:
            d = data["dof_positions"].shape[-1]
            print(f"dof_positions D = {d} | dof_names count = {len(dof_names)}")
            if d != len(dof_names):
                print("⚠️ Mismatch: dof_positions last-dim != number of dof_names")

        if dof_names and "dof_velocities" in keys:
            d = data["dof_velocities"].shape[-1]
            print(f"dof_velocities D = {d} | dof_names count = {len(dof_names)}")
            if d != len(dof_names):
                print("⚠️ Mismatch: dof_velocities last-dim != number of dof_names")

        if body_names and "body_positions" in keys:
            b = data["body_positions"].shape[-2]
            print(f"body_positions B = {b} | body_names count = {len(body_names)}")
            if b != len(body_names):
                print("⚠️ Mismatch: body_positions bodies != number of body_names")

        print("\n--- STRUCTURE ANALYSIS ---")
        if "motion" in keys or "pose" in keys:
            print("-> Seems to contain raw poses (Raw Motion).")
        if "obs" in keys or "amp_obs" in keys:
            print("-> Seems to contain pre-calculated observations (Ready for AMP).")

    except Exception as e:
        print(f"ERROR during loading: {e}")


def verify_frame():
    print(f"--- Verifying Frames for: {FILE_PATH} ---")
    data = np.load(FILE_PATH, allow_pickle=True)

    # [T, N_bodies, 3]
    pos = torch.tensor(data["body_positions"], dtype=torch.float32)
    vel = torch.tensor(data["body_linear_velocities"], dtype=torch.float32)
    rot = torch.tensor(data["body_rotations"], dtype=torch.float32)  # quats

    fps = float(data["fps"])
    dt = 1.0 / fps

    vel_calc_world = (pos[1:] - pos[:-1]) / dt
    vel_stored = vel[:-1]

    diff_world = (vel_calc_world - vel_stored).abs().mean()

    root_rot = rot[:-1, 0, :]  # [T-1, 4]
    root_vel_calc_body = quat_rotate_inv(root_rot, vel_calc_world[:, 0, :])
    root_vel_stored = vel_stored[:, 0, :]

    diff_body = (root_vel_calc_body - root_vel_stored).abs().mean()

    print(f"\nIpotesi 1: Dati salvati in WORLD Frame")
    print(f"  Errore Medio (Finite Diff vs Stored): {diff_world:.4f}")

    print(f"\nIpotesi 2: Dati salvati in BODY Frame")
    print(f"  Errore Medio (Ruotando in Body):      {diff_body:.4f}")

    print("\n--- VERDETTO ---")
    if diff_world < diff_body:
        print("✅ I DATI SONO IN WORLD FRAME (Come previsto)")
        print("-> Devi convertirli in Body Frame nel Dataset!")
    else:
        print("⚠️ I DATI SONO GIÀ IN BODY FRAME")
        print("-> NON convertirli, usali così!")


if __name__ == "__main__":
    inspect_dataset()
    verify_frame()
