from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import validate_imitation_cfg, Activation
from ..residual_vq import ResidualVQ, RVQCfg
from .expert_policy import Decoder


class ImitationBlock(nn.Module):
    """
    Forward-only block (losses computed elsewhere in compute_losses.py):
      zp = prior(s)
      z  = posterior(s, goal)
      y  = z - sg(zp)
      y_hat = y (or RVQ(y))
      z_hat = sg(zp) + y_hat
      a_hat = decoder(s, z_hat)
    """

    def __init__(
        self,
        s_dim: int,
        goal_dim: int,
        action_dim: int,
        net_cfg: dict,
        expert_decoder: Decoder = None,
    ):
        super().__init__()
        self.s_dim = int(s_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)

        validate_imitation_cfg(net_cfg, self.s_dim, self.goal_dim, self.action_dim)

        self.prior = PriorNet(obs_dim=self.s_dim, env_cfg=net_cfg)
        self.posterior = PosteriorNet(
            obs_dim=self.s_dim, goal_dim=self.goal_dim, env_cfg=net_cfg
        )

        self.latent_dim = int(
            net_cfg["imitation_learning_policy"]["prior"]["units"][-1]
        )

        dec_cfg = net_cfg["imitation_learning_policy"]["action_decoder"]
        self.use_expert_rollouts = net_cfg["imitation_learning_policy"].get(
            "use_expert_rollouts", True
        )

        if self.use_expert_decoder:
            self.decoder = expert_decoder
            if self.decoder is None:
                raise ValueError(
                    "expert_decoder must be provided when use_expert_decoder is True"
                )
            self.decoder.eval()  # freeze decoder
            for param in self.decoder.parameters():
                param.requires_grad = False

        else:
            self.decoder = ActionDecoder(
                obs_dim=self.s_dim,
                latent_dim=self.latent_dim,
                action_dim=self.action_dim,
                cfg_dict=dec_cfg,
            )

        rvq_cfg = RVQCfg(
            dim=self.latent_dim,
            num_quantizers=int(
                net_cfg["imitation_learning_policy"]["rvq_cfg"].get("num_quantizers", 8)
            ),
            codebook_size=int(
                net_cfg["imitation_learning_policy"]["rvq_cfg"].get(
                    "codebook_size", 1024
                )
            ),
            quantize_dropout=bool(
                net_cfg["imitation_learning_policy"]["rvq_cfg"].get(
                    "quantize_dropout", True
                )
            ),
            codebook_dim=int(
                net_cfg["imitation_learning_policy"]["rvq_cfg"].get(
                    "codebook_dim", None
                )
            ),
            shared_codebook=bool(
                net_cfg["imitation_learning_policy"]["rvq_cfg"].get(
                    "shared_codebook", False
                )
            ),
            decay=float(
                net_cfg["imitation_learning_policy"]["rvq_cfg"].get("decay", 0.99)
            ),
            eps=float(net_cfg["imitation_learning_policy"]["rvq_cfg"].get("eps", 1e-5)),
            commitment_weight=float(
                net_cfg["imitation_learning_policy"]["rvq_cfg"].get(
                    "commitment_weight", 1.0
                )
            ),
            kmeans_init=bool(
                net_cfg["imitation_learning_policy"]["rvq_cfg"].get(
                    "kmeans_init", False
                )
            ),
            kmeans_iters=int(
                net_cfg["imitation_learning_policy"]["rvq_cfg"].get("kmeans_iters", 10)
            ),
        )
        self.rvq = ResidualVQ(rvq_cfg)

    @torch.no_grad()
    def get_action(self, s_cur: torch.Tensor) -> torch.Tensor:
        assert s_cur.shape[-1] == self.s_dim
        zp = self.prior(s_cur)
        return self.decoder(s_cur, zp)

    @staticmethod
    def compute_imitation_losses(
        out: Dict[str, torch.Tensor],
        a_expert_mu: torch.Tensor,
        weights: LossWeights,
        next_out: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        out expected keys:
        a_hat, zp, z, y_hat, z_hat
        optionally vq_info.loss_vq
        next_out (optional) keys:
        zp, y_hat

        Returns dict with:
        loss_total + individual losses

        This function implement the losses described in the paper section 4.3.1 formula (6).
        1) action reconstruction loss
        2) margin-minimization loss
        3) regularization loss (temporal consistency)
        4) Commitment loss (VQ loss)
        """
        a_hat = out["a_hat"]
        zp = out["zp"]  # Prior latent vector
        y_hat = out.get("y_hat", None)
        z_hat = out["z_hat"]  # Prior + quantized residual

        # action reconstruction loss
        loss_action = F.mse_loss(a_hat, a_expert_mu)
        # margin-minimization: push prior to be predictive / reduce residual energy
        loss_mm = F.mse_loss(z_hat.detach(), zp)

        # regularization loss (temporal consistency): minimize changes between latent embeddings of neighboring frames.
        loss_reg = torch.zeros((), device=a_hat.device, dtype=a_hat.dtype)
        if weights.reg > 0.0 and next_out is not None:
            zp_next = next_out["zp"]
            y_hat_next = next_out.get("y_hat", None)
            if y_hat_next is None:
                raise ValueError(
                    "next_out must contain 'y_hat' when computing regularization loss."
                )
            loss_reg = F.mse_loss(zp, zp_next) + F.mse_loss(y_hat, y_hat_next)

        # vq loss default = 0 if not present
        loss_vq = torch.zeros((), device=a_hat.device, dtype=a_hat.dtype)
        vq_info = out.get("vq_info", None)
        if isinstance(vq_info, dict) and "loss_vq" in vq_info:
            loss_vq = vq_info["loss_vq"]

        loss_total = (
            weights.action * loss_action
            + weights.mm * loss_mm
            + weights.reg * loss_reg
            + weights.vq * loss_vq
        )

        return {
            "loss_total": loss_total,
            "loss_action": loss_action,
            "loss_mm": loss_mm,
            "loss_reg": loss_reg,
            "loss_commit": loss_vq,
        }

    def forward(
        self, s_cur: torch.Tensor, goal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Algorithm 1. Action prediction using MQ"""
        assert s_cur.shape[-1] == self.s_dim
        assert goal.shape[-1] == self.goal_dim

        zp = self.prior(s_cur)  # (B, latent)
        z = self.posterior(s_cur, goal)  # (B, latent)
        # Calculate margin with stop gradient
        zp_sg = zp.detach()
        y = z - zp_sg
        # Calculate y_hat and commitment loss via RVQ.
        # vq_info is a dict containing vq_loss.
        y_hat, vq_info = self.rvq(y)
        # Reconstruct z_hat
        z_hat = y_hat + zp_sg
        # Sample action
        a_hat = self.decoder(s_cur, z_hat)

        return {
            "a_hat": a_hat,
            "zp": zp,
            "z": z,
            "y": y,
            "y_hat": y_hat,
            "z_hat": z_hat,
            "vq_info": vq_info,
        }


@dataclass
class LossWeights:
    action: float = 1.0
    mm: float = 0.1
    reg: float = 0.0
    l2: float = 0.0
    vq: float = 1.0  # keep as 1; you can tune later


class PriorNet(nn.Module):
    """Prior: z_p = f_theta(s_cur)"""

    def __init__(self, obs_dim: int, env_cfg: dict):
        super().__init__()
        cfg = env_cfg["imitation_learning_policy"]["prior"]
        self.units = [int(u) for u in cfg["units"]]
        self.obs_dim = int(obs_dim)
        self.act_name = str(cfg["activation"])
        self._build_net()

    def _build_net(self):
        layers = []
        in_size = self.obs_dim
        for i in range(len(self.units) - 1):
            h = self.units[i]
            layers.append(nn.Linear(in_size, h))
            layers.append(Activation(self.act_name))
            in_size = h
        layers.append(nn.Linear(in_size, self.units[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, s_cur: torch.Tensor) -> torch.Tensor:
        return self.net(s_cur)


class PosteriorNet(nn.Module):
    """Posterior: z = f_phi(s_cur, goal)"""

    def __init__(self, obs_dim: int, goal_dim: int, env_cfg: dict):
        super().__init__()
        cfg = env_cfg["imitation_learning_policy"]["posterior"]
        self.units = [int(u) for u in cfg["units"]]
        self.obs_dim = int(obs_dim)
        self.goal_dim = int(goal_dim)
        self.act_name = str(cfg["activation"])
        self._build_net()

    def _build_net(self):
        layers = []
        in_size = self.obs_dim + self.goal_dim
        for i in range(len(self.units) - 1):
            h = self.units[i]
            layers.append(nn.Linear(in_size, h))
            layers.append(Activation(self.act_name))
            in_size = h
        layers.append(nn.Linear(in_size, self.units[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, s_cur: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s_cur, goal], dim=-1)
        return self.net(x)


class ActionDecoder(nn.Module):
    """a_hat = pi_low(s_cur, z_hat)"""

    def __init__(self, obs_dim: int, latent_dim: int, action_dim: int, cfg_dict: dict):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.act_name = str(cfg_dict["activation"])
        self.hidden_units = [int(u) for u in cfg_dict["hidden_units"]]
        self._build_net()

    def _build_net(self):
        layers = []
        in_size = self.obs_dim + self.latent_dim
        for h in self.hidden_units:
            layers.append(nn.Linear(in_size, h))
            layers.append(Activation(self.act_name))
            in_size = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_size, self.action_dim)

    def forward(self, s_cur: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s_cur, z_hat], dim=-1)
        x = self.net(x)
        return self.mu_head(x)
