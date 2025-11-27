"""Similar to trainer, but with student-teacher"""

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .schema import EpochHistory, create_empty_history, record_epoch
from .trainer import AsyncCheckpointer
from .utils import plot_training_curves

logger = logging.getLogger("training")


def _get_unwrapped_model(model: nn.Module) -> nn.Module:
    """Get the original model from a potentially torch.compile'd model."""
    # torch.compile wraps the model and stores original in _orig_mod
    return getattr(model, "_orig_mod", model)


class DistillationLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.3,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hard_loss = self.ce(student_logits, labels)

        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T**2)

        total = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total, {"hard": hard_loss, "soft": soft_loss, "total": total}


class DistillationTrainer:
    """Mirrors Trainer but uses student-teacher distillation."""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineAnnealingLR | StepLR,
        device: torch.device,
        run_dir: Path,
        temperature: float = 4.0,
        alpha: float = 0.3,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        early_stopping_patience: int = 5,
        gradient_accumulation_steps: int = 1,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_dir = run_dir
        self.grad_clip = grad_clip
        self.early_stopping_patience = early_stopping_patience
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.criterion = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            class_weights=class_weights.to(device) if class_weights is not None else None,
        )

        self.checkpointer = AsyncCheckpointer()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.history: EpochHistory = create_empty_history()
        self._interrupted = False
        self._current_epoch = 0
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        def handler(signum: int, frame: Any) -> None:
            if self._interrupted:
                sys.exit(1)
            logger.warning("\nInterrupt! Saving checkpoint...")
            self._interrupted = True

        signal.signal(signal.SIGINT, handler)

    def _save_emergency_checkpoint(self) -> None:
        unwrapped = _get_unwrapped_model(self.student)
        checkpoint = {
            "epoch": self._current_epoch,
            "model_state_dict": unwrapped.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "interrupted": True,
        }
        torch.save(checkpoint, self.run_dir / "emergency_checkpoint.pth")
        logger.info("Emergency checkpoint saved")

    def _get_teacher_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Get teacher predictions (no grad, optionally with AMP)."""
        with torch.no_grad():
            if self.use_amp:
                with autocast("cuda", dtype=torch.float16):
                    logits: torch.Tensor = self.teacher(images)
                    return logits
            logits = self.teacher(images)
            return logits

    def train_epoch(self, train_loader: DataLoader[Any], epoch: int) -> dict[str, float]:
        self.student.train()
        total_loss, total_hard, total_soft = 0.0, 0.0, 0.0
        num_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            if self._interrupted:
                break

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            teacher_logits = self._get_teacher_logits(images)

            if self.use_amp and self.scaler is not None:
                with autocast("cuda", dtype=torch.float16):
                    student_logits = self.student(images)
                    loss, ld = self.criterion(student_logits, teacher_logits, labels)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                student_logits = self.student(images)
                loss, ld = self.criterion(student_logits, teacher_logits, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.gradient_accumulation_steps
            total_hard += ld["hard"].item()
            total_soft += ld["soft"].item()
            num_batches += 1

            pbar.set_postfix(loss=f"{loss.item() * self.gradient_accumulation_steps:.4f}")

        n = max(num_batches, 1)
        return {"loss": total_loss / n, "hard": total_hard / n, "soft": total_soft / n}

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader[Any]) -> dict[str, float]:
        self.student.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds: list[int] = []
        all_labels: list[int] = []

        pbar = tqdm(val_loader, desc="Eval", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels_dev = labels.to(self.device, non_blocking=True)

            teacher_logits = self._get_teacher_logits(images)

            if self.use_amp:
                with autocast("cuda", dtype=torch.float16):
                    student_logits = self.student(images)
                    loss, _ = self.criterion(student_logits, teacher_logits, labels_dev)
            else:
                student_logits = self.student(images)
                loss, _ = self.criterion(student_logits, teacher_logits, labels_dev)

            total_loss += loss.item()
            num_batches += 1
            all_preds.extend(student_logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_accuracy": float(accuracy_score(all_labels, all_preds)),
            "val_f1": float(f1_score(all_labels, all_preds, average="macro")),
        }

    def _save_checkpoint(self, epoch: int, val_f1: float, is_best: bool) -> None:
        unwrapped = _get_unwrapped_model(self.student)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": unwrapped.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_f1": val_f1,
            "history": self.history,
        }
        self.checkpointer.save(checkpoint, str(self.run_dir / "latest_checkpoint.pth"))
        if is_best:
            self.checkpointer.save(checkpoint, str(self.run_dir / "best_model.pth"))

    def load_checkpoint(self, path: Path | str) -> int:
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        unwrapped = _get_unwrapped_model(self.student)
        unwrapped.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        epoch: int = checkpoint["epoch"]
        logger.info(f"Resumed from epoch {epoch} (F1={checkpoint.get('val_f1', 0):.4f})")
        return epoch + 1

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        num_epochs: int,
        start_epoch: int = 1,
    ) -> dict[str, Any]:
        best_f1 = max(self.history["val_f1"]) if self.history["val_f1"] else 0.0
        best_epoch = self.history["val_f1"].index(best_f1) + 1 if self.history["val_f1"] else 0
        patience_counter = 0
        start_time = time.time()

        logger.info(
            f"Distillation: {num_epochs} epochs (from {start_epoch}), device={self.device}, AMP={self.use_amp}"
        )

        try:
            for epoch in range(start_epoch, num_epochs + 1):
                self._current_epoch = epoch
                if self._interrupted:
                    self._save_emergency_checkpoint()
                    break

                train_metrics = self.train_epoch(train_loader, epoch)
                if self._interrupted:
                    self._save_emergency_checkpoint()
                    break

                val_metrics = self.evaluate(val_loader)
                current_lr = self.optimizer.param_groups[0]["lr"]

                record_epoch(
                    self.history,
                    train_loss=train_metrics["loss"],
                    val_loss=val_metrics["val_loss"],
                    val_f1=val_metrics["val_f1"],
                    val_accuracy=val_metrics["val_accuracy"],
                    lr=current_lr,
                )

                logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Loss: {train_metrics['loss']:.4f} (hard={train_metrics['hard']:.4f}, soft={train_metrics['soft']:.4f}), "
                    f"Val F1: {val_metrics['val_f1']:.4f}, LR: {current_lr:.6f}"
                )

                is_best = val_metrics["val_f1"] > best_f1
                if is_best:
                    best_f1, best_epoch, patience_counter = val_metrics["val_f1"], epoch, 0
                    logger.info(f"  -> Best F1: {best_f1:.4f}")
                else:
                    patience_counter += 1

                self._save_checkpoint(epoch, val_metrics["val_f1"], is_best)
                self.scheduler.step()

                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        except Exception as e:
            logger.error(f"Error: {e}")
            self._save_emergency_checkpoint()
            raise

        training_time = time.time() - start_time
        logger.info(
            f"Done! Best F1: {best_f1:.4f} @ epoch {best_epoch}, time: {training_time:.1f}s"
        )

        self.checkpointer.wait()
        self.checkpointer.shutdown()
        plot_training_curves(self.history, self.run_dir)

        best_path = self.run_dir / "best_model.pth"
        if best_path.exists():
            unwrapped = _get_unwrapped_model(self.student)
            unwrapped.load_state_dict(torch.load(best_path, weights_only=False)["model_state_dict"])

        return {
            "best_epoch": best_epoch,
            "best_f1": best_f1,
            "training_time_seconds": training_time,
            "history": self.history,
            "interrupted": self._interrupted,
        }
