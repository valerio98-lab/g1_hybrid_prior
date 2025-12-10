"""
This script performs low-level training sanity checks for the LowLevelExpertPolicy model
using the G1HybridPriorDataset.

The tests include:
- Verify that the dataset and the model speak the same language (shapes, device, etc.).
- Check that the network can overfit a small portion of the data.
- Inspect the distributions of ground-truth vs predicted actions.
- Test that obs/goal are actually used (by shuffling the goal).
"""

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from g1_hybrid_prior.dataset import G1HybridPriorDataset
from g1_hybrid_prior.expert_policy import LowLevelExpertPolicy
from g1_hybrid_prior.helpers import get_project_root            


def build_batch_tensors(batch: Dict[str, torch.Tensor], device: torch.device):
    """
    Costruisce obs, goal, actions_gt a partire dal dict del DataLoader.

    Scelte (per debug):
    - obs  = [root_pos, root_quat_wxyz, joints, root_lin_vel, root_ang_vel, joint_vel]
    - goal = joints
    - actions_gt = joints (joint targets come "expert actions")
    """
    root_pos = batch["root_pos"]        # (B, 3)
    root_quat = batch["root_quat_wxyz"]  # (B, 4)
    joints = batch["joints"]          # (B, dof)
    root_lin_vel = batch["root_lin_vel"]    # (B, 3)
    root_ang_vel= batch["root_ang_vel"]    # (B, 3)
    joint_vel = batch["joint_vel"]       # (B, dof)

    # obs completo: stato + velocità
    obs = torch.cat(
        [
            root_pos,
            root_quat,
            joints,
            root_lin_vel,
            root_ang_vel,
            joint_vel,
        ],
        dim=-1,
    )  

    # per questo debug usiamo joints sia come goal che come azione target
    goal = joints.clone()        # (B, dof)
    actions_gt = joints.clone()  # (B, dof)

    obs = obs.to(device)
    goal = goal.to(device)
    actions_gt = actions_gt.to(device)

    return obs, goal, actions_gt


def compute_action_stats(name: str, actions: torch.Tensor):
    """Stampa statistiche base su un tensore di azioni (B, action_dim)."""
    with torch.no_grad():
        mean = actions.mean(dim=0)
        std = actions.std(dim=0)
        min_val, _ = actions.min(dim=0)
        max_val, _ = actions.max(dim=0)

        print(f"\n[{name}] stats (per joint) - primi 5 joint:")
        print("  mean[0:5]:", mean[:5].cpu().numpy())
        print("  std [0:5]:", std[:5].cpu().numpy())
        print("  min [0:5]:", min_val[:5].cpu().numpy())
        print("  max [0:5]:", max_val[:5].cpu().numpy())


def sanity_check_batch_shapes(dataset: G1HybridPriorDataset, device: torch.device):
    """Controlla shape, NaN e costruzione obs/goal/actions su un singolo batch."""
    print("\n=== Sanity check batch shapes ===")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    obs, goal, actions_gt = build_batch_tensors(batch, device=device)
    print(f"\n  obs.shape:        {obs.shape}")
    print(f"  goal.shape:       {goal.shape}")
    print(f"  actions_gt.shape: {actions_gt.shape}")

    # check NaN / inf
    assert torch.isfinite(obs).all(), "NaN/Inf in obs"
    assert torch.isfinite(goal).all(), "NaN/Inf in goal"
    assert torch.isfinite(actions_gt).all(), "NaN/Inf in actions_gt"
    print("Nessun NaN/Inf in obs/goal/actions_gt.")


def overfit_mini_subset(
    dataset: G1HybridPriorDataset,
    cfg: dict,
    device: torch.device,
    num_samples: int = 256,
    epochs: int = 20,
    lr: float = 3e-4,
):
    """
    Test: il modello riesce a overfittare una mini-porzione del dataset?
    Se no, quasi sicuramente c'è un bug nel wiring.
    """
    print("\n=== Overfit mini-subset test ===")

    # Costruiamo una mini-subset
    indices = list(range(min(num_samples, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=0)

    # Recuperiamo una batch per determinare le dimensioni
    first_batch = next(iter(loader))
    obs_ex, goal_ex, actions_gt_ex = build_batch_tensors(first_batch, device=device)

    obs_dim = obs_ex.shape[-1]
    goal_dim = goal_ex.shape[-1]
    action_dim = actions_gt_ex.shape[-1]

    print(f"  obs_dim = {obs_dim}, goal_dim = {goal_dim}, action_dim = {action_dim}")

    # Costruiamo il modello
    policy = LowLevelExpertPolicy(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        cfg=cfg,
        device=device,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        policy.train()
        running_loss = 0.0
        num_batches = 0

        for batch in loader:
            obs, goal, actions_gt = build_batch_tensors(batch, device=device)

            optimizer.zero_grad()
            mu_pred = policy(obs, goal)
            loss = mse_loss(mu_pred, actions_gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        print(f"  [Epoch {epoch:02d}] overfit mini-subset - loss: {avg_loss:.6f}")

    # Analisi finale su un batch
    policy.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        obs, goal, actions_gt = build_batch_tensors(batch, device=device)
        mu_pred = policy(obs, goal)
        loss = mse_loss(mu_pred, actions_gt).item()
        diff = mu_pred - actions_gt
        joint_mse = (diff ** 2).mean(dim=0)  # per joint

        print(f"\n  Final mini-subset loss: {loss:.6f}")
        print("  MSE per joint (primi 5):", joint_mse[:5].cpu().numpy())

        compute_action_stats("GT actions (mini-subset)", actions_gt)
        compute_action_stats("Pred actions (mini-subset)", mu_pred)

    print("=== Fine overfit mini-subset ===\n")


def train_full_debug(
    dataset: G1HybridPriorDataset,
    cfg: dict,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 3e-4,
    log_interval: int = 20,
):
    """
    Mini training "full dataset" per vedere trend della loss e qualche diagnostica extra.
    """
    print("\n=== Full-dataset debug training ===")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Determiniamo le dimensioni da una batch
    first_batch = next(iter(loader))
    obs_ex, goal_ex, actions_gt_ex = build_batch_tensors(first_batch, device=device)
    obs_dim = obs_ex.shape[-1]
    goal_dim = goal_ex.shape[-1]
    action_dim = actions_gt_ex.shape[-1]

    print(f"  obs_dim = {obs_dim}, goal_dim = {goal_dim}, action_dim = {action_dim}")

    # Ricostruiamo il loader (il primo iter l'abbiamo consumato)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    policy = LowLevelExpertPolicy(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        cfg=cfg,
        device=device,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    global_step = 0
    for epoch in range(1, epochs + 1):
        policy.train()
        running_loss = 0.0
        num_batches = 0

        for batch in loader:
            obs, goal, actions_gt = build_batch_tensors(batch, device=device)

            optimizer.zero_grad()
            mu_pred = policy(obs, goal)
            loss = mse_loss(mu_pred, actions_gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                with torch.no_grad():
                    diff = mu_pred - actions_gt
                    batch_mse = (diff ** 2).mean().item()
                    print(
                        f"  [Epoch {epoch:02d} | step {global_step:05d}] "
                        f"loss = {loss.item():.6f}, batch MSE = {batch_mse:.6f}"
                    )

        avg_loss = running_loss / max(1, num_batches)
        print(f"  >>> [Epoch {epoch:02d}] avg loss: {avg_loss:.6f}")

    # Diagnostica su un batch finale
    policy.eval()
    with torch.no_grad():
        loader_eval = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        batch = next(iter(loader_eval))
        obs, goal, actions_gt = build_batch_tensors(batch, device=device)
        mu_pred = policy(obs, goal)
        loss = mse_loss(mu_pred, actions_gt).item()
        diff = mu_pred - actions_gt
        joint_mse = (diff ** 2).mean(dim=0)

        print(f"\n  Final eval loss (full dataset): {loss:.6f}")
        print("  MSE per joint (primi 5):", joint_mse[:5].cpu().numpy())

        compute_action_stats("GT actions (full batch)", actions_gt)
        compute_action_stats("Pred actions (full batch)", mu_pred)

        # Test: la rete usa davvero il goal?
        perm = torch.randperm(goal.shape[0], device=device)
        goal_shuffled = goal[perm]
        mu_pred_shuffled = policy(obs, goal_shuffled)
        loss_shuffled = mse_loss(mu_pred_shuffled, actions_gt).item()

        print(
            f"\n  Loss con goal normale:   {loss:.6f}\n"
            f"  Loss con goal shufflato: {loss_shuffled:.6f}"
        )
        print("  (Idealmente la loss con goal shufflato dovrebbe essere più alta.)")

    print("=== Fine full-dataset debug training ===\n")


def main():
    parser = argparse.ArgumentParser(description="Mini cinematic debug training for LowLevelExpertPolicy.")
    root = get_project_root()

    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(root / "data_raw" / "LAFAN1_Retargeting_Dataset" / "g1" / "dance1_subject1.csv"),
        help="Percorso al CSV del dataset retargeted.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="g1",
        help="Nome del robot definito in robots.yaml (default: g1).",
    )
    parser.add_argument(
        "--epochs_mini",
        type=int,
        default=20,
        help="Epoch per l'overfit mini-subset.",
    )
    parser.add_argument(
        "--epochs_full",
        type=int,
        default=5,
        help="Epoch per il full-dataset debug training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size per il training full.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    # parser.add_argument(
    #     "--lazy_load",
    #     action="store_true",
    #     help="Usa lazy loading nel dataset.",
    # )
    # parser.add_argument(
    #     "--lazy_load_window",
    #     type=int,
    #     default=1000,
    #     help="Finestra per il lazy loading.",
    # )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cuda' o 'cpu'.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Carichiamo config di rete (network.yaml)
    cfg_path = root / "config" / "network.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Dataset
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")

    dataset = G1HybridPriorDataset(
        file_path=csv_path,
        robot=args.robot,
        lazy_load=False,
    )

    print(f"Dataset caricato: {len(dataset)} frames totali.")

    # 1) Sanity check shape + NaN
    sanity_check_batch_shapes(dataset, device=device)

    # 2) Overfit mini subset
    overfit_mini_subset(
        dataset=dataset,
        cfg=cfg,
        device=device,
        num_samples=256,
        epochs=args.epochs_mini,
        lr=args.lr,
    )

    # 3) Full dataset debug training
    train_full_debug(
        dataset=dataset,
        cfg=cfg,
        device=device,
        epochs=args.epochs_full,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=20,
    )


if __name__ == "__main__":
    main()
