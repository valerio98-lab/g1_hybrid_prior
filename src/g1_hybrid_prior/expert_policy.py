import torch
import torch.nn as nn
import yaml

from .helpers import get_project_root


class LowLevelExpertPolicy(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        cfg_path = get_project_root() / "config" / "network.yaml"
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
            torch.full((action_dim,), init_log_std, dtype=torch.float32), requires_grad=not sigma_fixed
        )  # learnable log standard deviation. PPO can use this directly during training.
        # Obv for a fixed sigma requires_grad=False

        self.to(self.device)

    def forward(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        target = target.to(self.device)
        z = self.encoder(obs, target)
        mu = self.decoder(obs, z)
        value = self.critic(obs, target)

        log_std = self.log_std.expand_as(mu)
        log_std_min = self.cfg_ll.get("log_std_min", -5.0)
        log_std_max = self.cfg_ll.get("log_std_max", 1.0)
        log_std = torch.clamp(log_std, log_std_min, log_std_max)
        return mu, log_std, value


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


# if __name__ == "__main__":
#     import yaml
#     from .helpers import get_project_root

#     cfg_path = get_project_root() / "config" / "network.yaml"
#     with open(cfg_path, "r") as f:
#         cfg = yaml.safe_load(f)
#     policy = LowLevelExpertPolicy(obs_dim=30, goal_dim=10, action_dim=29, cfg=cfg)
#     dummy_obs = torch.randn(4, 30)
#     dummy_goal = torch.randn(4, 10)
#     mu, log_std, value = policy(dummy_obs, dummy_goal)
#     print("mu shape:", mu.shape)
#     print("log_std shape:", log_std.shape)
#     print("value shape:", value.shape)
