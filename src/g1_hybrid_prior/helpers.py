
from pathlib import Path
import torch
import math


def get_project_root():
    return Path(__file__).resolve().parent.parent.parent


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi

def quat_normalize(q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_inv(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quat_log(q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    q = quat_normalize(q, eps)
    w = q[..., :1]
    v = q[..., 1:]
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    theta = torch.atan2(v_norm, w.clamp(-1+1e-8, 1-1e-8))
    axis = v / v_norm
    return 2.0 * theta * axis  
