import torch
import torch.nn as nn

class LowLevelExpertPolicy(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, cfg, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        latent_dim = cfg["low_level_expert_policy"]["encoder"]["units"][-1]

        self.encoder = Encoder(in_dim=obs_dim + goal_dim, cfg=cfg)
        self.decoder = Decoder(obs_dim=obs_dim, latent_dim=latent_dim, action_dim=action_dim, cfg=cfg)

        self.to(self.device)

    def forward(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs, target)   
        mu = self.decoder(obs, z)       
        return mu



class Encoder(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        enc_cfg = cfg["low_level_expert_policy"]["encoder"]
        hidden_units = enc_cfg["units"]
        act_name = enc_cfg["activation"]
        self.activation_fn = getattr(torch, act_name)

        layers = []
        prev = in_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            prev = h
        self.layers = nn.ModuleList(layers)

    def forward(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, target], dim=-1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation_fn(x)
        # ultimo layer lineare il latent non va schiacciato con la RELU
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
        for h in hidden_units:
            layers.append(nn.Linear(in_size + latent_dim, h))
            in_size = h
        self.layers = nn.ModuleList(layers)

        self.mu_head = nn.Linear(in_size, action_dim)

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = obs
        for layer in self.layers:
            aug = torch.cat([x, latent], dim=-1)  # [features, z]
            x = self.activation_fn(layer(aug))
        mu = self.mu_head(x)   # (B, action_dim)
        return mu



# if __name__ == "__main__":
#     from .helpers import get_project_root 

#     cfg_path = get_project_root() / "config" / "network.yaml"
#     import yaml
#     with open(cfg_path, "r") as f:
#         cfg = yaml.safe_load(f)
#     obs_dim = 30
#     goal_dim = 10
#     action_dim = 12
#     policy = LowLevelExpertPolicy(obs_dim, goal_dim, action_dim, cfg)  
#     dummy_obs = torch.randn(4, obs_dim)
#     dummy_goal = torch.randn(4, goal_dim)
#     actions = policy(dummy_obs, dummy_goal)
#     print("Actions shape:", actions.shape)  



        