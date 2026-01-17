# g1_hybrid_prior/imitation_trainer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainerCfg:
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    use_amp: bool = False

    log_every: int = 50

    # checkpointing
    ckpt_dir: str = "runs/ckpts"
    ckpt_every: int = 5_000
    keep_last_k: int = 5
    save_best: bool = False  # optional
    best_metric: str = "loss_total"  # metric name from eval stats
    best_mode: str = "min"  # "min" or "max"

    device: str = "cuda"


class ImitationTrainer:
    """
    Trainer agnostico: non sa nulla dell'env.
    batch richiesto:
      batch["s"]        (B, s_dim)
      batch["goal"]     (B, goal_dim)
      batch["a_expert"] (B, action_dim)
    opzionali (per reg temporal):
      batch["s_next"], batch["goal_next"]
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

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # optimizer: SOLO parametri trainabili
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

        self.scaler = torch.cuda.amp.GradScaler(
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

        # best checkpoint tracking (optional)
        self.best_value: Optional[float] = None
        self.best_path = self.ckpt_dir / "ckpt_best.pt"

    # -------------------------
    # Public API
    # -------------------------

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()

        s = batch["s"].to(self.device)
        goal = batch["goal"].to(self.device)
        a_expert = batch["a_expert"].to(self.device)

        s_next = batch.get("s_next", None)
        goal_next = batch.get("goal_next", None)
        if s_next is not None:
            s_next = s_next.to(self.device)
        if goal_next is not None:
            goal_next = goal_next.to(self.device)

        self.optim.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            out = self.model(s, goal)

            next_out = None
            if (
                s_next is not None
                and goal_next is not None
                and getattr(self.loss_weights, "reg", 0.0) > 0.0
            ):
                next_out = self.model(s_next, goal_next)

            losses_t = self.model.compute_imitation_losses(
                out=out,
                a_expert_mu=a_expert,
                weights=self.loss_weights,
                next_out=next_out,
            )

            loss_total = losses_t["loss_total"]

        # backward + step
        if self.scaler.is_enabled():
            self.scaler.scale(loss_total).backward()
            self._maybe_clip_grads()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss_total.backward()
            self._maybe_clip_grads()
            self.optim.step()

        self.global_step += 1

        stats = self._stats_from_out_and_losses(out, losses_t)

        # logging
        if self.writer is not None and (self.global_step % self.cfg.log_every == 0):
            self._tb_log(stats, prefix="train")

        # checkpointing
        self._maybe_save_checkpoint(train_stats=stats)

        return stats

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()

        s = batch["s"].to(self.device)
        goal = batch["goal"].to(self.device)
        a_expert = batch["a_expert"].to(self.device)

        s_next = batch.get("s_next", None)
        goal_next = batch.get("goal_next", None)
        if s_next is not None:
            s_next = s_next.to(self.device)
        if goal_next is not None:
            goal_next = goal_next.to(self.device)

        out = self.model(s, goal)

        next_out = None
        if (
            s_next is not None
            and goal_next is not None
            and getattr(self.loss_weights, "reg", 0.0) > 0.0
        ):
            next_out = self.model(s_next, goal_next)

        losses_t = self.model.compute_imitation_losses(
            out=out,
            a_expert_mu=a_expert,
            weights=self.loss_weights,
            next_out=next_out,
        )

        stats = self._stats_from_out_and_losses(out, losses_t)

        if self.writer is not None:
            self._tb_log(stats, prefix="eval")

        # optional "best ckpt" on eval
        if self.cfg.save_best:
            self._maybe_update_best(stats)

        return stats

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "cfg": self.cfg.__dict__,
        }
        torch.save(payload, str(path))

    def load(self, path: str | Path, strict: bool = True) -> None:
        payload = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(payload["model"], strict=strict)
        self.optim.load_state_dict(payload["optim"])
        if payload.get("scaler", None) is not None and self.scaler is not None:
            self.scaler.load_state_dict(payload["scaler"])
        self.global_step = int(payload.get("global_step", 0))

    # -------------------------
    # Internals
    # -------------------------

    def _maybe_clip_grads(self) -> None:
        if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(
                self._trainable_params(), self.cfg.grad_clip_norm
            )

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
            if "num_active" in vq_info:
                stats["rvq/num_active"] = float(vq_info["num_active"].detach().item())
            if "loss_vq" in vq_info:
                stats["rvq/loss_vq"] = float(vq_info["loss_vq"].detach().item())

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

    def _maybe_save_checkpoint(self, train_stats: Dict[str, float]) -> None:
        if self.cfg.ckpt_every <= 0:
            return
        if self.global_step % self.cfg.ckpt_every != 0:
            return

        ckpt_path = self.ckpt_dir / f"ckpt_{self.global_step:09d}.pt"
        self.save(ckpt_path)
        self._rotate_checkpoints()

        # optionale: best anche su train (io lo lascio su eval di default)
        # if self.cfg.save_best:
        #     self._maybe_update_best(train_stats)

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

    def _maybe_update_best(self, eval_stats: Dict[str, float]) -> None:
        metric = self.cfg.best_metric
        if metric not in eval_stats:
            return

        value = float(eval_stats[metric])

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
