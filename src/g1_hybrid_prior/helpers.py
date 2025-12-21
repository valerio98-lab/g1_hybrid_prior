
from pathlib import Path
import torch
import math


def get_project_root():
    return Path(__file__).resolve().parent.parent.parent


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """
    Wraps the input tensor values to the range [-pi, pi].
    """
    return (x + math.pi) % (2 * math.pi) - math.pi

def quat_normalize(q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Normalizes a quaternion tensor to have unit norm.
    """
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiplies two quaternion tensors.
    """
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """
    Computes the inverse of a quaternion tensor.
    """
    w, x, y, z = q.unbind(-1)
    conj = torch.stack([w, -x, -y, -z], dim=-1)
    norm_2 = (q * q).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return conj / norm_2

def quat_canonical(q: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion tensor to its canonical form (non-negative scalar part).
    """
    q = quat_normalize(q)
    mask = (q[..., :1] < 0)
    return torch.where(mask, -q, q)

def quat_log(q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Computes the logarithm of a quaternion tensor.
    """
    q = quat_canonical(q)
    w = q[..., :1].clamp(-1.0 + 1e-8, 1.0 - 1e-8)
    v = q[..., 1:]
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    theta = torch.atan2(v_norm, w)
    axis = v / v_norm
    rotvec_big = 2.0 * theta * axis
    rotvec_small = 2.0 * v
    small = (v_norm < 1e-4)
    return torch.where(small, rotvec_small, rotvec_big)


def rotate_world_to_body(v_world: torch.Tensor, q_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Rotates a vector from world coordinates to body coordinates using the given quaternion.
    """
    w, x, y, z = q_wxyz.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
    ], dim=-1).reshape(3,3)
    return R.t().matmul(v_world)  # (3,)


def quat_rotate(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) v by quaternion(s) q.
    q: (..., 4) in wxyz
    v: (..., 3)
    """
    q = quat_normalize(q_wxyz)
    w = q[..., :1]
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2.0 * (w * uv + uuv)

def quat_rotate_inv(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_inv(q_wxyz), v)

def quat_log_vec(q_wxyz: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Returns rotation-vector (theta * axis) corresponding to quaternion q.
    (Your current quat_log returns this already; I’m just making it explicit.)
    """
    q = quat_normalize(q_wxyz, eps)
    w = q[..., :1].clamp(-1.0 + 1e-8, 1.0 - 1e-8)
    v = q[..., 1:]
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    theta = torch.atan2(v_norm, w)
    axis = v / v_norm
    return 2.0 * theta * axis
