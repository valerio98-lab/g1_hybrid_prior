# g1_hybrid_prior/models/task_learning_block.py
"""
Task-Learning Block (Paper Section 4.3.2).

Architecture:
  - Frozen: prior network, RVQ codebooks, low-level decoder
  - Trainable: high-level policy (categorical over codebook indices)

Forward flow:
  1. zp = prior(s)                          [frozen]
  2. indices = high_level(s, g_task)         [trainable, categorical]
  3. y_bar = sum of selected codebook entries[frozen codebooks]
  4. z_bar = y_bar + zp
  5. action = decoder(s, z_bar)             [frozen]
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import yaml

from ..helpers import get_project_root
from ..residual_vq import ResidualVQ
from ..utils import Activation
from .hybrid_imitation_block import ImitationBlock, PriorNet, ActionDecoder


class TaskLearningBlock(nn.Module):
    """
    Complete task-learning module.

    Loads a trained ImitationBlock checkpoint, freezes everything,
    and exposes a trainable HighLevelPolicy that selects codebook indices.

    The forward pass implements Figure 3 (right):
      s → Prior [frozen] → zp
      (s, g_task) → HighLevel [trainable] → categorical → indices
      indices → Codebook lookup [frozen] → y_bar
      z_bar = y_bar + zp
      (s, z_bar) → Decoder [frozen] → action
    """

    def __init__(
        self,
        s_dim: int,
        goal_dim: int,
        task_goal_dim: int,
        action_dim: int,
        imitation_ckpt_path: str,
        expert_ckpt_path: str,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.goal_dim = goal_dim
        self.task_goal_dim = task_goal_dim
        self.action_dim = action_dim

        cfg_path = get_project_root() / "config/TaskLearning.yaml"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.num_active_codebooks = cfg["task_learning_policy"]["num_active_codebooks"]
        use_expert_decoder = cfg.get("imitation_learning_policy", {}).get(
            "use_expert_decoder", False
        )

        if use_expert_decoder:
            from .expert_policy import ExpertPolicy

            expert = ExpertPolicy(
                obs_dim=s_dim, goal_dim=goal_dim, action_dim=action_dim
            )
            expert.load_from_rlgames(expert_ckpt_path, strict=False, load_rms=False)
            expert_decoder = expert.decoder
        else:
            expert_decoder = None

        self.imitation = ImitationBlock(
            s_dim=s_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            expert_decoder=expert_decoder,
        )

        full_obs_dim = s_dim + goal_dim
        self.obs_normalizer = StaticRunningMeanStd(shape=(full_obs_dim,))

        self._load_imitation_checkpoint_and_normalization_stats(
            imitation_ckpt_path, expert_ckpt_path
        )
        self._freeze_imitation()

        # Extract codebook info
        self.latent_dim = self.imitation.latent_dim
        self.codebook_size = self.imitation.rvq.cfg.codebook_size
        self.total_codebooks = self.imitation.rvq.num_quantizers

        num_active_codebooks = cfg["task_learning_policy"]["num_active_codebooks"]

        assert (
            num_active_codebooks <= self.total_codebooks
        ), f"num_active_codebooks={num_active_codebooks} > total={self.total_codebooks}"

        self.high_level = HighLevelPolicy(
            s_dim=s_dim,
            goal_dim=task_goal_dim,
            codebook_size=self.codebook_size,
            num_active_codebooks=num_active_codebooks,
        )

    def _load_imitation_checkpoint_and_normalization_stats(
        self, ckpt_path_imi: str, ckpt_path_exp: str
    ):
        """Load imitation block weights from checkpoint."""
        print(f"[TaskLearningBlock] Loading imitation checkpoint: {ckpt_path_imi}")
        ckpt = torch.load(ckpt_path_imi, map_location="cpu", weights_only=False)

        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = self.imitation.load_state_dict(state_dict, strict=False)
        print(
            f"[TaskLearningBlock] Imitation weights loaded. "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print(f"Missing keys (first 10): {missing[:10]}")
        if unexpected:
            print(f"Unexpected keys (first 10): {unexpected[:10]}")

        print(f"[TaskLearningBlock] Loading normalization stats from: {ckpt_path_exp}")
        ckpt = torch.load(ckpt_path_exp, map_location="cpu", weights_only=False)
        model_state = ckpt.get("model", ckpt)
        mean_key = next((k for k in model_state if "running_mean" in k), None)
        var_key = next((k for k in model_state if "running_var" in k), None)

        if mean_key and var_key:
            print(f"  Found RMS keys: {mean_key}, {var_key}")
            self.obs_normalizer.mean.copy_(model_state[mean_key])
            self.obs_normalizer.var.copy_(model_state[var_key])
        else:
            print(
                "[WARNING] Could not find running_mean/var in checkpoint! Zeros/Ones used."
            )

    def _freeze_imitation(self):
        """Freeze all imitation block parameters (prior, posterior, decoder, RVQ)."""
        for param in self.imitation.parameters():
            param.requires_grad = False
        self.imitation.eval()
        print("[TaskLearningBlock] Imitation block frozen.")

    def _normalize_s(self, s_raw):
        B = s_raw.shape[0]
        dummy_goal = torch.zeros(
            (B, self.goal_dim), device=s_raw.device, dtype=s_raw.dtype
        )

        full_raw = torch.cat([s_raw, dummy_goal], dim=-1)
        full_norm = self.obs_normalizer(full_raw)
        s_norm = full_norm[..., : self.s_dim]
        return s_norm

    def _lookup_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up codebook entries and sum them (RVQ reconstruction).

        Args:
            indices: (B, num_active_codebooks) — integer indices

        Returns:
            y_bar: (B, latent_dim) — sum of selected code vectors
        """
        B = indices.shape[0]
        y_bar = torch.zeros(B, self.latent_dim, device=indices.device)

        for q in range(self.num_active_codebooks):
            # Access the q-th codebook's embedding
            vq_layer = self.imitation.rvq.layers[q]
            codebook = vq_layer._codebook.embed  # (1, K, D) or (K, D)
            if codebook.dim() == 3:
                codebook = codebook.squeeze(0)  # (K, D)
            else:
                raise AttributeError(
                    f"Cannot find codebook embeddings in VQ layer {q}. "
                    f"Available attrs: {[a for a in dir(vq_layer) if not a.startswith('__')]}"
                )

            idx_q = indices[:, q]  # (B,)
            selected = codebook[idx_q]  # (B, D)
            y_bar = y_bar + selected

        y_bar = self.imitation.rvq.project_out(y_bar)
        return y_bar

    def forward(
        self, s: torch.Tensor, g_task: torch.Tensor, deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for task learning.

        Args:
            s: (B, s_dim) — proprioceptive state
            g_task: (B, task_goal_dim) — task-specific goal (e.g., velocity command)
            deterministic: if True, use argmax instead of sampling

        Returns:
            dict with: action, logits, indices, log_prob, zp, z_bar, y_bar
        """
        s_norm = self._normalize_s(s)
        with torch.no_grad():
            zp = self.imitation.prior(s_norm)  # (B, latent_dim)

        hl_out = self.high_level(s_norm, g_task)
        logits = hl_out["logits"]  # (B, num_active_codebooks, codebook_size)

        if deterministic:
            indices = logits.argmax(dim=-1)  # (B, num_active_codebooks)
            # Still compute log_prob for debugging/analysis, even though it won't be used
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(indices)  # (B, num_active_codebooks)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            indices = dist.sample()  # (B, num_active_codebooks)
            log_prob = dist.log_prob(indices)  # (B, num_active_codebooks)

        # Sum log_probs across codebooks for total action log_prob
        log_prob_total = log_prob.sum(dim=-1)  # (B,)
        entropy = dist.entropy().sum(dim=-1)  # (B,)

        with torch.no_grad():
            y_bar = self._lookup_codebook(indices)  # (B, latent_dim)

        z_bar = y_bar + zp  # zp already detached since prior is frozen

        with torch.no_grad():
            action = self.imitation.decoder(s, z_bar)  # (B, action_dim)

        return {
            "action": action,
            "logits": logits,
            "indices": indices,
            "log_prob": log_prob_total,
            "entropy": entropy,
            "zp": zp,
            "z_bar": z_bar,
            "y_bar": y_bar,
        }

    def get_action_and_value(
        self,
        s: torch.Tensor,
        g_task: torch.Tensor,
        value_net: Optional[nn.Module] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method for PPO: returns action + value + log_prob + entropy.
        value_net is an external critic (not part of this block).
        """
        out = self.forward(s, g_task, deterministic=deterministic)

        if value_net is not None:
            value = value_net(s, g_task)
            out["value"] = value

        return out

    def evaluate_actions(
        self,
        s: torch.Tensor,
        g_task: torch.Tensor,
        old_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        For PPO training phase: recompute log_prob and entropy for old actions.

        Args:
            s: (B, s_dim)
            g_task: (B, task_goal_dim)
            old_indices: (B, num_active_codebooks) — indices from rollout

        Returns:
            log_prob, entropy, and the reconstructed action
        """

        with torch.no_grad():
            zp = self.imitation.prior(s)

        hl_out = self.high_level(s, g_task)
        logits = hl_out["logits"]  # (B, num_active_codebooks, codebook_size)

        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(old_indices)  # (B, num_active_codebooks)
        log_prob_total = log_prob.sum(dim=-1)  # (B,)
        entropy = dist.entropy().sum(dim=-1)  # (B,)

        with torch.no_grad():
            y_bar = self._lookup_codebook(old_indices)
            z_bar = y_bar + zp
            action = self.imitation.decoder(s, z_bar)

        return {
            "log_prob": log_prob_total,
            "entropy": entropy,
            "action": action,
            "logits": logits,
        }


class HighLevelPolicy(nn.Module):
    """
    High-level policy: π_high(s, g) -> categorical distribution over codebook indices.

    For each active codebook, outputs a categorical distribution over codebook_size entries.
    During training (PPO), we sample from the categorical; during inference we can take argmax.

    Architecture:
      MLP trunk -> one linear head per active codebook -> Categorical per codebook
    """

    def __init__(
        self,
        s_dim: int,
        goal_dim: int,
        codebook_size: int,
        num_active_codebooks: int = 1,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.goal_dim = goal_dim
        self.codebook_size = codebook_size
        self.num_active_codebooks = num_active_codebooks

        cfg_path = get_project_root() / "config/TaskLearning.yaml"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        hidden_units = cfg["task_learning_policy"]["high_level_policy"]["units"]
        activation = cfg["task_learning_policy"]["high_level_policy"]["activation"]
        self.activation_fn = Activation(activation)

        layers = []
        in_size = s_dim + goal_dim
        for h in hidden_units:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.activation_fn)
            in_size = h
        self.trunk = nn.Sequential(*layers)

        # One classification head per active codebook
        self.heads = nn.ModuleList(
            [nn.Linear(in_size, codebook_size) for _ in range(num_active_codebooks)]
        )

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            logits: (B, num_active_codebooks, codebook_size)
        """
        x = torch.cat([s, g], dim=-1)
        h = self.trunk(x)

        logits_list = [head(h) for head in self.heads]  # list of (B, codebook_size)
        logits = torch.stack(
            logits_list, dim=1
        )  # (B, num_active_codebooks, codebook_size)

        return {"logits": logits}


"""
Value network (critic) for task learning PPO.
The Value network is used as far as we are training the high-level policy with PPO. 
After training, it can be discarded since we only care about the learned high-level policy and the frozen low-level decoder. 
"""


class TaskCritic(nn.Module):
    """
    V(s, g_task) -> scalar value estimate.
    """

    def __init__(
        self,
        s_dim: int,
        goal_dim: int,
    ):
        super().__init__()
        cfg_path = get_project_root() / "config/TaskLearning.yaml"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        hidden_units = cfg["task_learning_policy"]["critic"]["units"]

        activation = cfg["task_learning_policy"]["critic"].get("activation", "relu")
        self.activation_fn = Activation(activation)

        layers = []
        in_size = s_dim + goal_dim
        for h in hidden_units:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.activation_fn)
            in_size = h
        layers.append(nn.Linear(in_size, 1))

        self.critic = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, g], dim=-1)
        return self.critic(x)


class StaticRunningMeanStd(nn.Module):
    """Modulo di normalizzazione con statistiche fisse (caricate da checkpoint)."""

    def __init__(self, shape, epsilon=1e-4, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.epsilon = epsilon
        self.clip = clip

    def forward(self, x):
        # x: (..., D)
        scale = torch.rsqrt(self.var + self.epsilon)
        x_norm = (x - self.mean) * scale
        return torch.clamp(x_norm, -self.clip, self.clip)

    def load_from_dict(self, state_dict, prefix="running_"):
        # Helper per caricare da dizionari RL-Games
        if f"{prefix}mean" in state_dict:
            self.mean.copy_(state_dict[f"{prefix}mean"])
        if f"{prefix}var" in state_dict:
            self.var.copy_(state_dict[f"{prefix}var"])


##DEBUGGING##
# if __name__ == "__main__":
#     s_dim = 69
#     goal_dim = 69
#     task_goal_dim = 5
#     action_dim = 29
#     imitation_ckpt_path = "/home/valerio/g1_hybrid_gym/logs/imitation/22_01_2026_202836/ckpts/g1_hybrid_imitation/ckpt_best.pt"

#     block = TaskLearningBlock(
#         s_dim=s_dim,
#         goal_dim=goal_dim,
#         task_goal_dim=task_goal_dim,
#         action_dim=action_dim,
#         imitation_ckpt_path=imitation_ckpt_path,
#     )

#     batch_size = 4
#     s = torch.randn(batch_size, s_dim)
#     g_task = torch.randn(batch_size, task_goal_dim)

#     out = block(s, g_task, deterministic=True)
#     print("Output keys:", out.keys())
#     print("Action shape:", out["action"].shape)
