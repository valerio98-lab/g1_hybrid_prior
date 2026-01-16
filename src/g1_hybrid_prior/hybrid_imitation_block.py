# hybrid_imitation_block.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Activation(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        name = name.lower()
        if name == "relu":
            self.act = nn.ReLU()
        elif name == "tanh":
            self.act = nn.Tanh()
        elif name in ("silu", "swish"):
            self.act = nn.SiLU()
        elif name == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation '{name}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@dataclass
class ImitationLossWeights:
    action: float = 1.0
    mm: float = 0.1
    reg: float = 0.0  # off by default (needs temporal pairs)
    l2: float = 0.0  # optional weight decay-like term if you want


class PriorNet(nn.Module):
    """
    Prior: z_p = f_theta(s_cur)
    """

    def __init__(self, obs_dim: int, env_cfg: dict):
        super().__init__()
        cfg = env_cfg["imitation_learning_policy"]["prior"]
        self.units = cfg["units"]
        self.obs_dim = obs_dim
        self.activation = _Activation(cfg["activation"])

        self._build_net()

    def _build_net(self):
        layers = []
        in_size = self.obs_dim
        for h in range(self.units - 1):
            layers.append(nn.Linear(in_size, self.units[h]))
            layers.append(self.activation)
            in_size = self.units[h]
        layers.append(nn.Linear(in_size, self.units[-1]))
        self.prior = nn.Sequential(*layers)

    def forward(self, s_cur: torch.Tensor) -> torch.Tensor:
        return self.prior(s_cur)


class PosteriorNet(nn.Module):
    """
    Posterior: z = f_phi(s_cur, goal)
    Here "goal" is your engineered target info (e.g. ref-diff features).
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        env_cfg: dict,
    ):
        super().__init__()
        cfg = env_cfg["imitation_learning_policy"]["posterior"]
        self.units = cfg["units"]
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.activation = _Activation(cfg["activation"])

    def _build_net(self):
        layers = []
        in_size = self.obs_dim + self.goal_dim
        for h in range(self.units - 1):
            layers.append(nn.Linear(in_size, self.units[h]))
            layers.append(self.activation)
            in_size = self.units[h]
        layers.append(nn.Linear(in_size, self.units[-1]))
        self.posterior = nn.Sequential(*layers)

    def forward(self, s_cur: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s_cur, goal], dim=-1)
        return self.posterior(x)


class ActionDecoder(nn.Module):
    """
    Low-level head: a_hat = pi_low(s_cur, z_hat)
    Mirrors your 'Decoder' style: concatenate [features, latent] at each layer.
    """

    def __init__(
        self,
        s_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden: Tuple[int, ...] = (512, 1024, 1024),
        activation: str = "relu",
    ):
        super().__init__()
        self.activation = _Activation(activation)

        layers = []
        in_size = s_dim
        for h in hidden:
            layers.append(nn.Linear(in_size + latent_dim, h))
            in_size = h
        self.layers = nn.ModuleList(layers)
        self.mu_head = nn.Linear(in_size, action_dim)

    def forward(self, s_cur: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        x = s_cur
        for layer in self.layers:
            x = self.activation(layer(torch.cat([x, z_hat], dim=-1)))
        mu = self.mu_head(x)
        return mu


class ImitationBlock(nn.Module):
    """
    Full imitation block without quantizer:
      z_p = prior(s)
      z   = posterior(s, goal)
      y   = z - sg(z_p)
      y_hat = y (placeholder for RVQ later)
      z_hat = sg(z_p) + y_hat
      a_hat = decoder(s, z_hat)

    Losses:
      L_action = MSE(a_hat, a_expert_mu)
      L_mm     = MSE(z_hat, z_p)   (with z_hat built from sg(z_p) -> grads push only prior)
      L_reg    = optional temporal smoothness (requires prev-step tensors)
    """

    def __init__(
        self,
        s_dim: int,
        goal_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        prior_hidden: Tuple[int, ...] = (1024, 1024),
        post_hidden: Tuple[int, ...] = (1024, 1024),
        dec_hidden: Tuple[int, ...] = (512, 1024, 1024),
        activation: str = "relu",
        loss_weights: Optional[ImitationLossWeights] = None,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.prior = PriorNet(s_dim, latent_dim, prior_hidden, activation)
        self.posterior = PosteriorNet(
            s_dim, goal_dim, latent_dim, post_hidden, activation
        )
        self.decoder = ActionDecoder(
            s_dim, latent_dim, action_dim, dec_hidden, activation
        )

        self.w = loss_weights or ImitationLossWeights()

    @torch.no_grad()
    def infer_action(self, s_cur: torch.Tensor) -> torch.Tensor:
        """
        Test-time style: uses only prior (no goal).
        NOTE: without RVQ, this is only a placeholder and may be weak early on.
        """
        zp = self.prior(s_cur)
        z_hat = zp  # if you want "pure prior" inference
        return self.decoder(s_cur, z_hat)

    def forward_train(
        self,
        s_cur: torch.Tensor,
        goal: torch.Tensor,
        a_expert_mu: Optional[torch.Tensor] = None,
        # Optional temporal inputs for L_reg (you can pass them later from rollout):
        s_cur_next: Optional[torch.Tensor] = None,
        goal_next: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with:
          - a_hat, z, zp, y, z_hat
          - losses: loss_total and individual scalars (if a_expert_mu given)
        """
        # --- Forward ---
        zp = self.prior(s_cur)  # (B, latent)
        z = self.posterior(s_cur, goal)  # (B, latent)

        zp_sg = zp.detach()
        y = z - zp_sg  # (B, latent)

        # Placeholder for RVQ: y_hat = y
        y_hat = y

        z_hat = zp_sg + y_hat  # (B, latent)

        a_hat = self.decoder(s_cur, z_hat)  # (B, action)

        out: Dict[str, torch.Tensor] = {
            "a_hat": a_hat,
            "zp": zp,
            "z": z,
            "y": y,
            "y_hat": y_hat,
            "z_hat": z_hat,
        }

        # --- Losses ---
        losses: Dict[str, torch.Tensor] = {}

        if a_expert_mu is not None:
            loss_action = F.mse_loss(a_hat, a_expert_mu)

            # Margin-minimizing loss:
            # With z_hat built from sg(zp), this pushes only zp (the "center") to minimize residual energy.
            loss_mm = F.mse_loss(z_hat, zp)

            loss_reg = torch.tensor(0.0, device=s_cur.device)
            if self.w.reg > 0.0:
                # Needs temporal pairs. We define a simple smoothness:
                #   ||zp_t - zp_{t+1}||^2 + ||y_t - y_{t+1}||^2
                # Note: in the paper they use reg on z_p and y_hat (or y_bar).
                if (s_cur_next is not None) and (goal_next is not None):
                    zp_next = self.prior(s_cur_next)
                    z_next = self.posterior(s_cur_next, goal_next)
                    y_next = z_next - zp_next.detach()
                    loss_reg = F.mse_loss(zp, zp_next) + F.mse_loss(y, y_next)
                else:
                    # You asked for no plumbing yet, so keep it safe.
                    loss_reg = torch.tensor(0.0, device=s_cur.device)

            loss_l2 = torch.tensor(0.0, device=s_cur.device)
            if self.w.l2 > 0.0:
                # optional: a tiny L2 on latents to prevent blow-up early
                loss_l2 = zp.pow(2).mean() + z.pow(2).mean() + y.pow(2).mean()

            loss_total = (
                self.w.action * loss_action
                + self.w.mm * loss_mm
                + self.w.reg * loss_reg
                + self.w.l2 * loss_l2
            )

            losses.update(
                {
                    "loss_total": loss_total,
                    "loss_action": loss_action,
                    "loss_mm": loss_mm,
                    "loss_reg": loss_reg,
                    "loss_l2": loss_l2,
                }
            )

        out["losses"] = losses
        return out
