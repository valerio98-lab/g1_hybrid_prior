import numpy as np
import torch

from g1_hybrid_prior.helpers import quat_rotate_inv

FILE_PATH = (
    "/home/valerio/g1_hybrid_prior/data_amp/LAFAN-G1/LAFAN_walk1_subject1_0_-1.npz"
)


def inspect_dataset():
    try:
        print(f"--- Inspecting: {FILE_PATH} ---")
        data = np.load(FILE_PATH, allow_pickle=True)

        keys = list(data.files)
        print(f"\n[KEYS FOUND]: {keys}")

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
                    if "quat" in k or obj.shape[-1] == 4:
                        norms = np.linalg.norm(obj, axis=-1)
                        print(
                            f"  -> Seems like a quaternion? "
                            f"Norm mean: {norms.mean():.4f}"
                        )

            else:
                print(f"\nKey: '{k}' -> Non-array type: {type(obj)}")

        print("\n--- STRUCTURE ANALYSIS ---")
        if "motion" in keys or "pose" in keys:
            print("-> Seems to contain raw poses (Raw Motion).")
        if "obs" in keys or "amp_obs" in keys:
            print("-> Seems to contain pre-calculated observations " "(Ready for AMP).")

    except Exception as e:
        print(f"ERROR during loading: {e}")


def verify_frame():
    print(f"--- Verifying Frames for: {FILE_PATH} ---")
    data = np.load(FILE_PATH, allow_pickle=True)

    # [T, N_bodies, 3]
    pos = torch.tensor(data["body_positions"], dtype=torch.float32)
    vel = torch.tensor(data["body_linear_velocities"], dtype=torch.float32)
    rot = torch.tensor(
        data["body_rotations"], dtype=torch.float32
    )  # Quaternioni (Root è ind 0)

    fps = float(data["fps"])
    dt = 1.0 / fps

    # 1. Calcoliamo la velocità tramite differenze finite sulle posizioni (WORLD)
    # v_calc = (p[t+1] - p[t]) / dt
    vel_calc_world = (pos[1:] - pos[:-1]) / dt

    # Confrontiamo con la velocità salvata (tagliando l'ultimo frame per matchare dimensioni)
    vel_stored = vel[:-1]

    # Calcolo Errore assumendo che 'vel_stored' sia WORLD
    diff_world = (vel_calc_world - vel_stored).abs().mean()

    # 2. Calcolo Errore assumendo che 'vel_stored' sia BODY (quindi proviamo a ruotare vel_calc in body)
    # Prendiamo la rotazione del root (o dei singoli body se sono locali al body, ma solitamente si intende root frame)
    # Per semplicità controlliamo il BODY 0 (ROOT)
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
