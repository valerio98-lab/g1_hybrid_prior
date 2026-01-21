# g1_hybrid_prior/imitation_trainer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainerCfg:
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    use_amp: bool = False
    best_ema_beta: float = 0.98

    log_every: int = 50

    # checkpointing
    ckpt_dir: str = "runs/ckpts"
    ckpt_every: int = 5_000
    keep_last_k: int = 5
    save_best: bool = True  # optional
    best_metric: str = "loss_action"  # metric name from eval stats
    best_mode: str = "min"  # "min" or "max"
    mm_warmup_steps: int = 0  # number of steps to warmup MM loss weight
    mm_start: float = 0.1  # starting weight for MM loss
    mm_end: float = 1.0  # final weight for MM loss

    device: str = "cuda"


@dataclass
class LossWeights:
    action: float = 1.0
    mm: float = 0.1
    reg: float = 0.05
    vq: float = 1.0  # keep as 1; you can tune later


class ImitationTrainer:
    """
    Trainer agnostico: non sa nulla dell'env.
    batch richiesto:
      batch["s"]        (B, s_dim)
      batch["goal"]     (B, goal_dim)
      batch["a_expert"] (B, action_dim)
    """

    def __init__(
        self,
        model: nn.Module,  # ImitationBlock
        loss_weights: Any,  # dataclass LossWeights
        cfg: TrainerCfg,
        log_dir: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        self.model = model
        self.loss_weights = loss_weights
        self.cfg = cfg

        self.best_ema: Optional[float] = None

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError(
                "No trainable parameters found (did you freeze everything by mistake?)."
            )

        self.optim = torch.optim.Adam(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        self.scaler = torch.amp.GradScaler(
            enabled=cfg.use_amp and self.device.type == "cuda"
        )

        self.global_step = 0

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None

        # checkpoint dir
        base = Path(cfg.ckpt_dir)
        if run_name is not None:
            base = base / run_name
        self.ckpt_dir = base
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_value: Optional[float] = None
        self.best_path = self.ckpt_dir / "ckpt_best.pt"

        self.prev_out = None
        self.prev_size = None

    def _masked_mse(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        x,y: (N, D)
        mask: (N,) bool or (N,1)
        returns scalar masked mean over N, averaged per-sample over D.
        """
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)
        mask_f = mask.to(dtype=x.dtype)
        # per-sample mse (N,1)
        per = ((x - y) ** 2).mean(dim=-1, keepdim=True)
        denom = mask_f.sum().clamp_min(1.0)
        return (per * mask_f).sum() / denom

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()

        s = batch["s"].to(self.device)
        goal = batch["goal"].to(self.device)
        a_expert = batch["a_expert"].to(self.device)

        valid_prev = batch.get("valid_prev", None)
        if valid_prev is not None:
            valid_prev = valid_prev.to(self.device, dtype=torch.bool)

        self.optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            enabled=self.scaler.is_enabled(), device_type=self.device.type
        ):
            out = self.model(s, goal)

            batch_size_zp = out["zp"].shape[0]
            latent_dim_zp = out["zp"].shape[1]
            if (self.prev_out is None) or (self.prev_size != batch_size_zp):
                self.prev_out = {
                    "zp": out["zp"].detach().clone(),
                    "y_hat": out["y_hat"].detach().clone(),
                }
                self.prev_size = batch_size_zp

            # inside train_step, before compute_imitation_losses
            if self.cfg.mm_warmup_steps > 0:
                t = min(1.0, self.global_step / float(self.cfg.mm_warmup_steps))
                self.loss_weights.mm = (
                    1.0 - t
                ) * self.cfg.mm_start + t * self.cfg.mm_end
            else:
                self.loss_weights.mm = self.cfg.mm_end

            losses_t = self.compute_imitation_losses(
                out=out,
                a_expert_mu=a_expert,
                weights=self.loss_weights,
                prev_out=self.prev_out,
                valid_prev=valid_prev,
            )

            loss_total = losses_t["loss_total"]
            self.prev_out["zp"] = out["zp"].detach().clone()
            self.prev_out["y_hat"] = out["y_hat"].detach().clone()

        # backward + step
        if self.scaler.is_enabled():
            self.scaler.scale(loss_total).backward()
            grad_norm = self._clip_grads()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss_total.backward()
            grad_norm = self._clip_grads()
            self.optim.step()

        self.global_step += 1

        stats = self._stats_from_out_and_losses(out, losses_t)
        stats["optim/grad_norm"] = grad_norm
        stats["optim/lr"] = float(self.optim.param_groups[0]["lr"])
        stats["weights/mm"] = float(self.loss_weights.mm)
        stats["sched/mm_t"] = (
            float(min(1.0, self.global_step / float(self.cfg.mm_warmup_steps)))
            if self.cfg.mm_warmup_steps > 0
            else 1.0
        )

        # logging
        if self.writer is not None and (self.global_step % self.cfg.log_every == 0):
            self._tb_log(stats, prefix="train")

        if self.cfg.save_best:
            self._update_best(stats=stats)

        # checkpointing
        self._save_checkpoint(train_stats=stats)

        return stats

    # @torch.no_grad()
    # def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    #     self.prev_out = None
    #     self.model.eval()

    #     s = batch["s"].to(self.device)
    #     goal = batch["goal"].to(self.device)
    #     a_expert = batch["a_expert"].to(self.device)

    #     out = self.model(s, goal)

    #     if self.prev_out is None:
    #         self.prev_out = {
    #             "zp": out["zp"].detach(),
    #             "y_hat": out["y_hat"].detach(),
    #         }

    #     losses_t = self.compute_imitation_losses(
    #         out=out,
    #         a_expert_mu=a_expert,
    #         weights=self.loss_weights,
    #         prev_out=self.prev_out,
    #     )
    #     self.prev_out = {
    #         "zp": out["zp"].detach(),
    #         "y_hat": out["y_hat"].detach(),
    #     }

    #     stats = self._stats_from_out_and_losses(out, losses_t)

    #     if self.writer is not None and (self.global_step % self.cfg.log_every == 0):
    #         self._tb_log(stats, prefix="eval")

    #     # optional "best ckpt" on eval
    #     if self.cfg.save_best:
    #         self._update_best(stats)

    #     return stats

    def _update_best(self, stats: Dict[str, float]) -> None:
        metric = self.cfg.best_metric
        if metric not in stats:
            return

        x = float(stats[metric])

        # EMA update
        if self.best_ema is None:
            self.best_ema = x
        else:
            b = float(self.cfg.best_ema_beta)
            b = max(0.0, min(0.9999, b))
            self.best_ema = b * self.best_ema + (1.0 - b) * x
        value = self.best_ema

        if self.best_value is None:
            better = True
        else:
            if self.cfg.best_mode == "min":
                better = value < self.best_value
            elif self.cfg.best_mode == "max":
                better = value > self.best_value
            else:
                raise ValueError("best_mode must be 'min' or 'max'")

        if better:
            self.best_value = value
            self.save(self.best_path)

    def compute_imitation_losses(
        self,
        out: Dict[str, torch.Tensor],
        a_expert_mu: torch.Tensor,
        weights: LossWeights,
        prev_out: Optional[Dict[str, torch.Tensor]] = None,
        valid_prev: Optional[torch.Tensor] = None,
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
        y_hat = out["y_hat"]
        z_hat = out["z_hat"]  # Prior + quantized residual

        # action reconstruction loss
        loss_action = F.mse_loss(a_hat, a_expert_mu)
        # margin-minimization: push prior to be predictive / reduce residual energy
        loss_mm = F.mse_loss(z_hat.detach(), zp)

        # regularization loss (temporal consistency): minimize changes between latent embeddings of neighboring frames.
        loss_reg = torch.zeros((), device=a_hat.device, dtype=a_hat.dtype)
        if weights.reg > 0.0:
            if prev_out is None or valid_prev is None:
                raise RuntimeError(
                    "Temporal reg enabled (weights.reg > 0) but batch['valid_prev'] or prev_out is missing."
                )
            loss_reg = self._masked_mse(
                out["zp"], prev_out["zp"], valid_prev
            ) + self._masked_mse(y_hat, prev_out["y_hat"], valid_prev)
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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "cfg": self.cfg.__dict__,
            "best_value": self.best_value,
            "best_ema": self.best_ema,
        }
        torch.save(payload, str(path))

    def load(self, path: str | Path, strict: bool = True) -> None:
        payload = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(payload["model"], strict=strict)
        self.optim.load_state_dict(payload["optim"])
        self.best_value = payload.get("best_value", None)
        self.best_ema = payload.get("best_ema", None)
        if payload.get("scaler", None) is not None and self.scaler is not None:
            self.scaler.load_state_dict(payload["scaler"])
        self.global_step = int(payload.get("global_step", 0))

    def _clip_grads(self) -> float:
        if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._trainable_params(), self.cfg.grad_clip_norm
            )
            return float(grad_norm.detach().item())
        return 0.0

    def _trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _tb_log(self, stats: Dict[str, float], prefix: str) -> None:
        assert self.writer is not None
        for k, v in stats.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)

    def _stats_from_out_and_losses(
        self, out: Dict[str, torch.Tensor], losses: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for k, v in losses.items():
            stats[k] = float(v.detach().item())

        # RVQ extras
        if "vq_info" in out and isinstance(out["vq_info"], dict):
            vq_info = out["vq_info"]

            # num_active + scalar loss
            if "num_active" in vq_info:
                num_active = int(vq_info["num_active"].detach().item())
                stats["rvq/num_active"] = float(num_active)
            else:
                num_active = 0

            if "loss_vq" in vq_info:
                stats["rvq/loss_vq"] = float(vq_info["loss_vq"].detach().item())

            # per-layer losses (commitment loss per quantizer)
            if "losses_per_layer" in vq_info:
                lpl = vq_info["losses_per_layer"].detach()
                # expected shape (K,) - robust if (B,K)
                if lpl.ndim == 2:
                    lpl = lpl.mean(dim=0)
                for i in range(lpl.shape[0]):
                    stats[f"rvq/loss_layer_{i}"] = float(lpl[i].item())

            # codebook usage + perplexity (per quantizer)
            if "indices" in vq_info:
                idx = vq_info["indices"].detach()  # (B, K_active)
                if idx.ndim == 2 and idx.numel() > 0:
                    B, K = idx.shape

                    # try to get codebook_size from model.rvq.cfg
                    codebook_size = 0
                    rvq = getattr(self.model, "rvq", None)
                    if rvq is not None and hasattr(rvq, "cfg"):
                        codebook_size = int(getattr(rvq.cfg, "codebook_size", 0))

                    perplexities = []

                    for q in range(K):
                        uq = torch.unique(idx[:, q]).numel()
                        stats[f"rvq/unique_codes_q{q}"] = float(uq)

                        if codebook_size > 0:
                            usage_ratio = uq / codebook_size
                            stats[f"rvq/usage_ratio_q{q}"] = float(usage_ratio)
                            stats[f"rvq/dead_fraction_q{q}"] = float(1.0 - usage_ratio)
                            counts = torch.bincount(
                                idx[:, q], minlength=codebook_size
                            ).float()

                        else:
                            counts = torch.bincount(idx[:, q]).float()

                        p = counts / (counts.sum() + 1e-8)
                        entropy = -(p * (p + 1e-8).log()).sum()
                        perplexity = torch.exp(entropy)
                        stats[f"rvq/perplexity_q{q}"] = float(perplexity.item())
                        perplexities.append(perplexity)

                    # mean perplexity across active quantizers
                    if len(perplexities) > 0:
                        stats["rvq/perplexity_mean"] = float(
                            torch.stack(perplexities).mean().item()
                        )
                        stats["rvq/perplexity_min"] = float(
                            torch.stack(perplexities).min().item()
                        )
                        stats["rvq/perplexity_max"] = float(
                            torch.stack(perplexities).max().item()
                        )

        # latent norms (debug)
        if "y_hat" in out:
            stats["latent/y_hat_norm"] = float(
                out["y_hat"].detach().norm(dim=-1).mean().item()
            )
        if "zp" in out:
            stats["latent/zp_norm"] = float(
                out["zp"].detach().norm(dim=-1).mean().item()
            )
        if "z_hat" in out:
            stats["latent/z_hat_norm"] = float(
                out["z_hat"].detach().norm(dim=-1).mean().item()
            )

        return stats

    def _save_checkpoint(self, train_stats: Dict[str, float]) -> None:
        if self.cfg.ckpt_every <= 0:
            return
        if self.global_step % self.cfg.ckpt_every != 0:
            return

        ckpt_path = self.ckpt_dir / f"ckpt_{self.global_step:09d}.pt"
        self.save(ckpt_path)
        self._rotate_checkpoints()

    def _rotate_checkpoints(self) -> None:
        k = int(self.cfg.keep_last_k)
        if k <= 0:
            return

        ckpts = sorted(self.ckpt_dir.glob("ckpt_*.pt"))
        if len(ckpts) <= k:
            return

        to_delete = ckpts[:-k]
        for p in to_delete:
            try:
                p.unlink()
            except OSError:
                pass
