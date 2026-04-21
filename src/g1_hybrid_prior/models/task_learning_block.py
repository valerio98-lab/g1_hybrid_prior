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

from typing import Dict, Optional

import torch
import torch.nn as nn

import yaml

from ..helpers import get_project_root
from ..utils import Activation
from .hybrid_imitation_block import ImitationBlock


class TaskLearningBlock(nn.Module):
    """
    Loads a frozen imitation learning model and adds a trainable high-level policy on top.
    The high-level policy picks codebook indices based on state and task goal.
    The frozen decoder then generates actions from these indices.
    """

    def __init__(
        self,
        s_dim: int,
        goal_dim: int,
        task_goal_dim: int,
        action_dim: int,
        imitation_ckpt_path: str,
        expert_ckpt_path: str,
        cfg_path: str,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.goal_dim = goal_dim
        self.task_goal_dim = task_goal_dim
        self.action_dim = action_dim

        # Load config to get number of active codebooks
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.num_active_codebooks = cfg["task_learning_policy"]["num_active_codebooks"]
        print(
            f"[TaskLearningBlock] num_active_codebooks={self.num_active_codebooks}",
            flush=True,
        )
        use_expert_decoder = cfg.get("imitation_learning_policy", {}).get(
            "use_expert_decoder", False
        )

        # Optionally load expert decoder
        if use_expert_decoder:
            from .expert_policy import ExpertPolicy

            expert = ExpertPolicy(
                obs_dim=s_dim, goal_dim=goal_dim, action_dim=action_dim
            )
            expert.load_from_rlgames(expert_ckpt_path, strict=False, load_rms=False)
            expert_decoder = expert.decoder
        else:
            expert_decoder = None

        # Create the imitation block (prior, decoder, codebooks)
        self.imitation = ImitationBlock(
            s_dim=s_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            expert_decoder=expert_decoder,
        )

        # Running mean/std for normalizing observations
        full_obs_dim = s_dim + goal_dim
        self.obs_normalizer = StaticRunningMeanStd(shape=(full_obs_dim,))

        # Load checkpoint and freeze everything
        self._load_imitation_checkpoint_and_normalization_stats(
            imitation_ckpt_path, expert_ckpt_path
        )
        self._freeze_imitation()

        # Get codebook dimensions from loaded imitation block
        self.latent_dim = self.imitation.latent_dim
        self.codebook_size = self.imitation.rvq.cfg.codebook_size
        self.total_codebooks = self.imitation.rvq.num_quantizers

        num_active_codebooks = cfg["task_learning_policy"]["num_active_codebooks"]

        assert (
            num_active_codebooks <= self.total_codebooks
        ), f"num_active_codebooks={num_active_codebooks} > total={self.total_codebooks}"

        # Create trainable high-level policy
        self.high_level = HighLevelPolicy(
            s_dim=s_dim,
            goal_dim=task_goal_dim,
            codebook_size=self.codebook_size,
            num_active_codebooks=num_active_codebooks,
        )

    def _load_imitation_checkpoint_and_normalization_stats(
        self, ckpt_path_imi: str, ckpt_path_exp: str
    ):
        """Load weights for the imitation block from checkpoint."""
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

        # Load normalization statistics from expert checkpoint
        print(f"[TaskLearningBlock] Loading normalization stats from: {ckpt_path_exp}")
        ckpt = torch.load(ckpt_path_exp, map_location="cpu", weights_only=False)
        model_state = ckpt.get("model", ckpt)
        mean_key = "running_mean_std.running_mean"
        var_key = "running_mean_std.running_var"

        if mean_key in model_state and var_key in model_state:
            mean_val = model_state[mean_key]
            var_val = model_state[var_key]

            expected_dim = self.s_dim + self.goal_dim
            if mean_val.shape[0] != expected_dim:
                raise RuntimeError(
                    f"RMS dim mismatch: checkpoint has {mean_val.shape[0]}, "
                    f"expected {expected_dim} (s_dim={self.s_dim}, goal_dim={self.goal_dim})"
                )

            self.obs_normalizer.mean.copy_(mean_val)
            self.obs_normalizer.var.copy_(var_val)
            print(
                f"  Loaded obs RMS: mean.mean={mean_val.mean():.4f}, var.mean={var_val.mean():.4f}"
            )
        else:
            print("[WARNING] Could not find running_mean_std in checkpoint!")

    def _freeze_imitation(self):
        """Stop gradients for all imitation block parameters."""
        for param in self.imitation.parameters():
            param.requires_grad = False
        self.imitation.eval()
        print("[TaskLearningBlock] Imitation block frozen.")

    def _normalize_s(self, s_raw):
        """Normalize state using loaded mean/std statistics."""
        B = s_raw.shape[0]
        # Create dummy goal to match expected input shape
        dummy_goal = torch.zeros(
            (B, self.goal_dim), device=s_raw.device, dtype=s_raw.dtype
        )

        full_raw = torch.cat([s_raw, dummy_goal], dim=-1)
        full_norm = self.obs_normalizer(full_raw)
        s_norm = full_norm[..., : self.s_dim]
        return s_norm

    def _lookup_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get codebook vectors for given indices and sum them up.

        Args:
                indices: (B, num_active_codebooks) — which codebook entry to use

        Returns:
                y_bar: (B, latent_dim) — summed codebook vectors
        """
        B = indices.shape[0]
        y_bar = torch.zeros(B, self.latent_dim, device=indices.device)

        # For each codebook, look up the vector and add it
        for q in range(self.num_active_codebooks):
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
        Run the full task learning forward pass.

        Args:
                s: (B, s_dim) — robot state
                g_task: (B, task_goal_dim) — task goal (e.g., velocity target)
                deterministic: if True, pick best indices instead of sampling

        Returns:
                dict with action, logits, indices, log_prob, entropy, etc.
        """
        # Normalize state
        s_norm = self._normalize_s(s)

        # Get prior latent (frozen)
        with torch.no_grad():
            zp = self.imitation.prior(s_norm)

        # High-level policy outputs logits over codebook indices
        hl_out = self.high_level(s_norm, g_task)
        logits = hl_out["logits"]  # (B, num_active_codebooks, codebook_size)

        # Either pick best or sample indices
        if deterministic:
            indices = logits.argmax(dim=-1)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(indices)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            indices = dist.sample()
            log_prob = dist.log_prob(indices)

        # Total log probability and entropy across all codebooks
        log_prob_total = log_prob.sum(dim=-1)  # (B,)
        entropy = dist.entropy().sum(dim=-1)  # (B,)

        # Look up codebook vectors and combine with prior
        with torch.no_grad():
            y_bar = self._lookup_codebook(indices)

        z_bar = y_bar + zp

        # Decode action (frozen)
        with torch.no_grad():
            action = self.imitation.decoder(s, z_bar)

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
        Get action, log_prob, entropy, and optionally value estimate.
        Used during PPO rollout.
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
        Re-evaluate old actions for PPO updates.
        Computes log_prob and entropy for actions taken during rollout.

        Args:
                s: (B, s_dim) — state
                g_task: (B, task_goal_dim) — task goal
                old_indices: (B, num_active_codebooks) — actions from rollout

        Returns:
                log_prob, entropy, reconstructed action
        """
        with torch.no_grad():
            zp = self.imitation.prior(s)

        # Get current policy logits
        hl_out = self.high_level(s, g_task)
        logits = hl_out["logits"]

        # Compute log_prob and entropy for old actions
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(old_indices)
        log_prob_total = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # Reconstruct action for reference
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
    Outputs a categorical distribution over codebook indices for each active codebook.
    Takes state and task goal as input.
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

        # Load network architecture from config
        cfg_path = get_project_root() / "config/TaskLearning.yaml"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        hidden_units = cfg["task_learning_policy"]["high_level_policy"]["units"]
        activation = cfg["task_learning_policy"]["high_level_policy"]["activation"]
        self.activation_fn = Activation(activation)

        # Build MLP trunk
        layers = []
        in_size = s_dim + goal_dim
        for h in hidden_units:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.activation_fn)
            in_size = h
        self.trunk = nn.Sequential(*layers)

        # One output head per codebook
        self.heads = nn.ModuleList(
            [nn.Linear(in_size, codebook_size) for _ in range(num_active_codebooks)]
        )

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute logits for each codebook.

        Returns:
                logits: (B, num_active_codebooks, codebook_size)
        """
        x = torch.cat([s, g], dim=-1)
        h = self.trunk(x)

        # Get logits from each head and stack them
        logits_list = [head(h) for head in self.heads]
        logits = torch.stack(logits_list, dim=1)

        return {"logits": logits}


class TaskCritic(nn.Module):
    """
    Value network for PPO training.
    Estimates V(s, g_task) — how good the current state is for achieving the task goal.
    """

    def __init__(
        self,
        s_dim: int,
        goal_dim: int,
    ):
        super().__init__()
        # Load architecture from config
        cfg_path = get_project_root() / "config/TaskLearning.yaml"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        hidden_units = cfg["task_learning_policy"]["critic"]["units"]
        activation = cfg["task_learning_policy"]["critic"].get("activation", "relu")
        self.activation_fn = Activation(activation)

        # Build MLP
        layers = []
        in_size = s_dim + goal_dim
        for h in hidden_units:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.activation_fn)
            in_size = h
        layers.append(nn.Linear(in_size, 1))

        self.critic = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimate for (state, goal) pair."""
        x = torch.cat([s, g], dim=-1)
        return self.critic(x)


class StaticRunningMeanStd(nn.Module):
    """
    Normalizes inputs using fixed mean and variance (loaded from checkpoint).
    Does not update statistics during training.
    """

    def __init__(self, shape, epsilon=1e-4, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.epsilon = epsilon
        self.clip = clip

    def forward(self, x):
        """Normalize and clip input."""
        scale = torch.rsqrt(self.var + self.epsilon)
        x_norm = (x - self.mean) * scale
        return torch.clamp(x_norm, -self.clip, self.clip)

    def load_from_dict(self, state_dict, prefix="running_"):
        """Load mean and variance from a state dict."""
        if f"{prefix}mean" in state_dict:
            self.mean.copy_(state_dict[f"{prefix}mean"])
        if f"{prefix}var" in state_dict:
            self.var.copy_(state_dict[f"{prefix}var"])
