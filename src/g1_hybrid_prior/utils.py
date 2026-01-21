from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn


def validate_imitation_cfg(
    env_cfg: dict, s_dim: int, goal_dim: int, action_dim: int
) -> None:
    assert (
        "imitation_learning_policy" in env_cfg
    ), "Missing env_cfg['imitation_learning_policy']"
    cfg = env_cfg["imitation_learning_policy"]

    assert (
        "prior" in cfg and "posterior" in cfg and "action_decoder" in cfg
    ), "imitation_learning_policy must contain prior/posterior/action_decoder"

    prior_units = list(cfg["prior"]["units"])
    post_units = list(cfg["posterior"]["units"])
    assert (
        len(prior_units) >= 1 and len(post_units) >= 1
    ), "prior/posterior units must be non-empty"

    prior_lat = int(prior_units[-1])
    post_lat = int(post_units[-1])
    assert (
        prior_lat == post_lat
    ), f"Final and first layers mismatch: prior[-1]={prior_lat} posterior[-1]={post_lat}"

    dec_cfg = cfg["action_decoder"]
    assert (
        "hidden_units" in dec_cfg and len(dec_cfg["hidden_units"]) > 0
    ), "action_decoder.hidden_units missing/empty"

    assert int(s_dim) > 0 and int(goal_dim) > 0 and int(action_dim) > 0


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save training checkpoint.

    Stored:
      - model state_dict
      - optimizer state_dict (optional)
      - global step (optional)
      - extra metadata (optional)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "step": step,
        "extra": extra,
    }

    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()

    torch.save(ckpt, path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Returns:
      dict with keys: step, extra
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return {
        "step": ckpt.get("step", None),
        "extra": ckpt.get("extra", None),
    }


class Activation(nn.Module):
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
