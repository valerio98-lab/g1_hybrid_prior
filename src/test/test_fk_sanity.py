import numpy as np
import pinocchio as pin
from pathlib import Path
from g1_hybrid_prior.robot_cfg import load_robot_cfg
from g1_hybrid_prior.data_preprocessor import EndEffectorAugmenter
from g1_hybrid_prior.helpers import get_project_root

# --- CONFIGURAZIONE TEST ---
# Modifica con i tuoi percorsi reali
URDF_PATH = str(get_project_root() / "assets" / "g1_29dof_with_hand_rev_1_0.urdf")
YAML_PATH = str(get_project_root() / "config" / "robots.yaml")

TARGET_EES = [
    "left_hand_palm_link", 
    "right_hand_palm_link", 
    "left_ankle_roll_link", 
    "right_ankle_roll_link"
]

def main():
    print("\n--- INIZIO TEST SANITY CHECK FK (FIXED) ---")
    
    try:
        cfg = load_robot_cfg(YAML_PATH, "g1")
        augmenter = EndEffectorAugmenter(URDF_PATH, TARGET_EES, cfg)
        print("✅ Inizializzazione Augmenter: OK")
    except Exception as e:
        print(f"❌ Errore Inizializzazione: {e}")
        return

    model = augmenter.model
    data = augmenter.data
    
    # 2. Test Posa Neutra
    q_neutral = np.zeros(model.nq)
    q_neutral[6] = 1.0  # Quat W = 1
    
    print("\n[TEST 1] Posa Neutra...")
    # --- FIX QUI ---
    pin.forwardKinematics(model, data, q_neutral)
    pin.updateFramePlacements(model, data)
    # ---------------
    
    neutral_pos = {}
    for i, name in enumerate(TARGET_EES):
        fid = augmenter.ee_frame_ids[i]
        pos = data.oMf[fid].translation
        neutral_pos[name] = pos.copy() # Importante fare .copy() per non sovrascrivere il riferimento
        print(f"   -> {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    ly = neutral_pos["left_hand_palm_link"][1]
    ry = neutral_pos["right_hand_palm_link"][1]
    
    if abs(ly + ry) < 0.05:
        print("   ✅ Simmetria OK")
    else:
        print(f"   ⚠️ WARNING: Asimmetria {ly} vs {ry}")

    # 3. Test Movimento Giunto Singolo
    print("\n[TEST 2] Movimento Spalla Sinistra (+45 deg)...")
    joint_name = "left_shoulder_pitch_joint"
    if model.existJointName(joint_name):
        jid = model.getJointId(joint_name)
        q_idx = model.joints[jid].idx_q
        
        q_test = q_neutral.copy()
        q_test[q_idx] = 0.785 
        
        # --- FIX QUI ---
        pin.forwardKinematics(model, data, q_test)
        pin.updateFramePlacements(model, data)
        # ---------------
        
        for i, name in enumerate(TARGET_EES):
            fid = augmenter.ee_frame_ids[i]
            curr_pos = data.oMf[fid].translation
            diff = np.linalg.norm(curr_pos - neutral_pos[name])
            
            if "left_hand" in name:
                if diff > 0.01:
                    print(f"   ✅ {name} si è mosso di {diff:.3f}m (Corretto)")
                else:
                    print(f"   ❌ {name} NON si è mosso! (Errore mapping)")
            elif "right_hand" in name or "ankle" in name:
                if diff < 0.001:
                    print(f"   ✅ {name} fermo (Corretto)")
                else:
                    print(f"   ❌ {name} si è mosso di {diff:.3f}m! (Errore isolamento)")

    # 4. Verifica Root Movement
    print("\n[TEST 3] Movimento Root (Base +1m su Z)...")
    q_root = q_neutral.copy()
    q_root[2] += 1.0 
    
    # --- FIX QUI ---
    pin.forwardKinematics(model, data, q_root)
    pin.updateFramePlacements(model, data)
    # ---------------
    
    fid_foot = augmenter.ee_frame_ids[2] # Left ankle
    pos_foot_raised = data.oMf[fid_foot].translation
    pos_foot_neutral = neutral_pos["left_ankle_roll_link"]
    
    z_diff = pos_foot_raised[2] - pos_foot_neutral[2]
    print(f"   -> Delta Z Piede: {z_diff:.3f}m")
    
    if abs(z_diff - 1.0) < 0.001:
        print("   ✅ La cinematica della Root Floating Base funziona")
    else:
        print(f"   ❌ Errore Root: atteso 1.0, ottenuto {z_diff}")

if __name__ == "__main__":
    main()