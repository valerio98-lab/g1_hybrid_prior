import torch
import torch.nn as nn
import yaml

from ..helpers import get_project_root

import math
from dataclasses import dataclass
from collections import OrderedDict


class RunningMeanStd(nn.Module):
    """
    Minimal RMS compatible with rl_games running_mean_std buffers:
      - running_mean_std.running_mean
      - running_mean_std.running_var
      - running_mean_std.count
    """

    def __init__(self, shape: int, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(1.0, dtype=torch.float32))

    def normalize(self, x: torch.Tensor, clip: float | None = None) -> torch.Tensor:
        # x: (..., D)
        mean = self.running_mean
        var = self.running_var
        x = (x - mean) / torch.sqrt(var + self.eps)
        if clip is not None:
            x = torch.clamp(x, -clip, clip)
        return x


class ExpertPolicy(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.obs_dim = int(obs_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)

        cfg_path = get_project_root() / "config" / "ExpertPPO.yaml"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        ll_cfg = cfg["low_level_expert_policy"]
        self.cfg_ll = ll_cfg

        latent_dim = cfg["low_level_expert_policy"]["encoder"]["units"][-1]

        self.encoder = Encoder(obs_dim=obs_dim, target_dim=goal_dim, cfg=cfg)
        self.decoder = Decoder(
            obs_dim=obs_dim, latent_dim=latent_dim, action_dim=action_dim, cfg=cfg
        )
        init_log_std = ll_cfg.get("init_log_std", -2.9)
        sigma_fixed = ll_cfg.get("sigma_fixed", True)

        self.critic = ValueHead(obs_dim=obs_dim, goal_dim=goal_dim, cfg=cfg)

        self.log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, dtype=torch.float32),
            requires_grad=not sigma_fixed,
        )  # learnable log standard deviation. PPO can use this directly
        # during training. Obv for a fixed sigma requires_grad=False

        self.apply_obs_rms = False  # IMPORTANT: keep False by default
        self.obs_rms = RunningMeanStd(shape=self.obs_dim + self.goal_dim)

        self.obs_clip = 5.0

        self.to(self.device)

    def enable_obs_norm(self, enabled: bool = True, clip: float | None = 5.0):
        self.apply_obs_rms = bool(enabled)
        self.obs_clip = clip

    def forward(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        target = target.to(self.device)

        if self.apply_obs_rms:
            full = torch.cat([obs, target], dim=-1)  # (B, obs_dim+goal_dim)
            full = self.obs_rms.normalize(full, clip=self.obs_clip)
            obs = full[..., : self.obs_dim]
            target = full[..., self.obs_dim :]

        z = self.encoder(obs, target)
        mu = self.decoder(obs, z)
        value = self.critic(obs, target)

        log_std = self.log_std.expand_as(mu)
        log_std_min = self.cfg_ll.get("log_std_min", -5.0)
        log_std_max = self.cfg_ll.get("log_std_max", 1.0)
        log_std = torch.clamp(log_std, log_std_min, log_std_max)
        return mu, log_std, value

    @torch.no_grad()
    def load_from_rlgames(
        self,
        ckpt_path: str,
        strict: bool = False,
        load_rms: bool = True,
        enable_rms: bool = True,
        clip: float | None = 5.0,
    ):
        """
        Loads:
          - a2c_network.* into this ExpertPolicy weights
          - (optional) running_mean_std.* into self.obs_rms
        This does NOT affect PPO training pipeline because PPO uses RL-Games wrapper normalization.
        Use this in play/inference when you instantiate ExpertPolicy standalone.
        """
        print(f"[INFO] Loading ExpertPolicy from rl_games checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

        # 1) load policy weights
        sd_model = {k: v for k, v in sd.items() if k.startswith("a2c_network.")}
        sd_model = OrderedDict(
            (k.replace("a2c_network.", "", 1), v) for k, v in sd_model.items()
        )
        missing, unexpected = self.load_state_dict(sd_model, strict=strict)
        print(
            f"[INFO] Expert weights loaded. missing={len(missing)} unexpected={len(unexpected)}"
        )

        # 2) load running_mean_std
        has_rms = (
            "running_mean_std.running_mean" in sd
            and "running_mean_std.running_var" in sd
            and "running_mean_std.count" in sd
        )
        print(f"[INFO] checkpoint has running_mean_std = {has_rms}")

        if load_rms and has_rms:
            mean = sd["running_mean_std.running_mean"].to(self.device).float()
            var = sd["running_mean_std.running_var"].to(self.device).float()
            count = sd["running_mean_std.count"].to(self.device).float()

            D = self.obs_dim + self.goal_dim
            if mean.numel() != D or var.numel() != D:
                raise RuntimeError(
                    f"[ExpertPolicy] RMS dim mismatch: checkpoint mean/var has {mean.numel()} but expected {D} "
                    f"(obs_dim={self.obs_dim}, goal_dim={self.goal_dim})."
                )

            self.obs_rms.running_mean.copy_(mean.view(-1))
            self.obs_rms.running_var.copy_(var.view(-1))
            self.obs_rms.count.copy_(count.view(()))

            self.enable_obs_norm(enable_rms, clip=clip)
            print(
                f"[INFO] Loaded running_mean_std into ExpertPolicy. apply_obs_rms={self.apply_obs_rms}"
            )

        return missing, unexpected


class Encoder(nn.Module):
    def __init__(self, obs_dim: int, target_dim: int, cfg: dict):
        super().__init__()
        enc_cfg = cfg["low_level_expert_policy"]["encoder"]
        hidden_units = enc_cfg["units"]
        act_name = enc_cfg["activation"]
        self.activation_fn = getattr(torch, act_name)

        layers = []
        prev = obs_dim + target_dim
        for unit in hidden_units:
            layers.append(nn.Linear(prev, unit))
            prev = unit
        self.layers = nn.ModuleList(layers)

    def forward(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, target], dim=-1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation_fn(
                    x
                )  # ultimo layer lineare il latent non va schiacciato con la RELU
        return x


class Decoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, action_dim: int, cfg: dict):
        super().__init__()
        dec_cfg = cfg["low_level_expert_policy"]["decoder"]
        hidden_units = dec_cfg["units"]
        act_name = dec_cfg["activation"]
        self.activation_fn = getattr(torch, act_name)

        layers = []
        in_size = obs_dim
        for unit in hidden_units:
            layers.append(nn.Linear(in_size + latent_dim, unit))
            in_size = unit
        self.layers = nn.ModuleList(layers)

        self.mu_head = nn.Linear(in_size, action_dim)

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = obs
        for layer in self.layers:
            aug = torch.cat([x, latent], dim=-1)  # [features, z]
            x = self.activation_fn(layer(aug))
        mu = self.mu_head(x)  # (B, action_dim)
        return mu


class ValueHead(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, cfg: dict):
        super().__init__()
        cfg = cfg["low_level_expert_policy"]["critic"]
        hidden_units = cfg["units"]
        act_name = cfg["activation"]
        if act_name.lower() == "relu":
            self.activation_fn = nn.ReLU()
        else:
            raise NotImplementedError(
                f"Activation {act_name} not implemented in ValueHead"
            )

        layers = []
        prev = obs_dim + goal_dim
        for unit in hidden_units:
            layers.append(nn.Linear(prev, unit))
            layers.append(self.activation_fn)
            prev = unit
        layers.append(nn.Linear(prev, 1))
        self.value_head = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, goal], dim=-1)
        value = self.value_head(x)

        return value


class LowLevelActor(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        cfg_path = get_project_root() / "config" / "ExpertPPO.yaml"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        ll_cfg = cfg["low_level_expert_policy"]
        self.cfg_ll = ll_cfg

        latent_dim = ll_cfg["encoder"]["units"][-1]

        self.encoder = Encoder(obs_dim=obs_dim, target_dim=goal_dim, cfg=cfg)
        self.decoder = Decoder(
            obs_dim=obs_dim, latent_dim=latent_dim, action_dim=action_dim, cfg=cfg
        )

        init_log_std = ll_cfg.get("init_log_std", -2.9)
        sigma_fixed = ll_cfg.get("sigma_fixed", True)

        self.log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, dtype=torch.float32),
            requires_grad=not sigma_fixed,
        )

        self.to(self.device)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor):
        obs = obs.to(self.device)
        goal = goal.to(self.device)

        z = self.encoder(obs, goal)
        mu = self.decoder(obs, z)

        log_std = self.log_std.expand_as(mu)
        log_std_min = self.cfg_ll.get("log_std_min", -5.0)
        log_std_max = self.cfg_ll.get("log_std_max", 1.0)
        log_std = torch.clamp(log_std, log_std_min, log_std_max)

        return mu, log_std


class LowLevelCritic(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        cfg_path = get_project_root() / "config" / "ExpertPPO.yaml"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.critic = ValueHead(obs_dim=obs_dim, goal_dim=goal_dim, cfg=cfg)
        self.to(self.device)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor):
        obs = obs.to(self.device)
        goal = goal.to(self.device)
        value = self.critic(obs, goal)
        return value


# if __name__ == "__main__":
#     import yaml
#     from .helpers import get_project_root

#     cfg_path = get_project_root() / "config" / "ExpertPPO.yaml"
#     with open(cfg_path, "r") as f:
#         cfg = yaml.safe_load(f)
#     policy = LowLevelExpertPolicy(obs_dim=30, goal_dim=10, action_dim=29, cfg=cfg)
#     dummy_obs = torch.randn(4, 30)
#     dummy_goal = torch.randn(4, 10)
#     mu, log_std, value = policy(dummy_obs, dummy_goal)
#     print("mu shape:", mu.shape)
#     print("log_std shape:", log_std.shape)
#     print("value shape:", value.shape)
