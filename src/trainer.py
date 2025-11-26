import logging
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import plot_training_curves

logger = logging.getLogger("training")


class AsyncCheckpointer:
    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[dict[str, Any], str] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            state_dict, path = item
            torch.save(state_dict, path)
            self._queue.task_done()

    def save(self, checkpoint: dict[str, Any], path: str) -> None:
        copied: dict[str, Any] = {}
        for k, v in checkpoint.items():
            if isinstance(v, dict):
                copied[k] = {
                    sk: sv.cpu().clone() if isinstance(sv, torch.Tensor) else sv
                    for sk, sv in v.items()
                }
            else:
                copied[k] = v
        self._queue.put((copied, path))

    def wait(self) -> None:
        self._queue.join()

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join()


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Literal["cosine", "step"],
    num_epochs: int,
) -> CosineAnnealingLR | StepLR:
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
    raise ValueError(f"Unknown scheduler: {scheduler_type}")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineAnnealingLR | StepLR,
        device: torch.device,
        run_dir: Path,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        early_stopping_patience: int = 5,
        gradient_accumulation_steps: int = 1,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_dir = run_dir
        self.grad_clip = grad_clip
        self.early_stopping_patience = early_stopping_patience
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
        self.checkpointer = AsyncCheckpointer()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "lr": [],
        }
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
        checkpoint = {
            "epoch": self._current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "interrupted": True,
        }
        torch.save(checkpoint, self.run_dir / "emergency_checkpoint.pth")
        logger.info("Emergency checkpoint saved")

    def train_epoch(self, train_loader: DataLoader[Any], epoch: int) -> float:
        self.model.train()
        total_loss, num_batches, acc_loss = 0.0, 0, 0.0
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            if self._interrupted:
                break

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp and self.scaler is not None:
                with autocast("cuda", dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels) / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                acc_loss += loss.item() * self.gradient_accumulation_steps

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels) / self.gradient_accumulation_steps
                loss.backward()
                acc_loss += loss.item() * self.gradient_accumulation_steps

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            total_loss += acc_loss
            num_batches += 1
            pbar.set_postfix({"loss": f"{acc_loss:.4f}"})
            acc_loss = 0.0

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader[Any]) -> dict[str, float]:
        self.model.eval()
        total_loss, num_batches = 0.0, 0
        all_preds, all_labels = [], []

        for images, labels in tqdm(val_loader, desc="Eval"):
            images = images.to(self.device, non_blocking=True)
            labels_dev = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with autocast("cuda", dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels_dev)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels_dev)

            total_loss += loss.item()
            num_batches += 1
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

        return {
            "val_loss": total_loss / num_batches,
            "val_accuracy": float(accuracy_score(all_labels, all_preds)),
            "val_f1": float(f1_score(all_labels, all_preds, average="macro")),
        }

    def _save_checkpoint(self, epoch: int, val_f1: float, is_best: bool) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
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
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        logger.info(
            f"Resumed from epoch {checkpoint['epoch']} (F1={checkpoint.get('val_f1', 0):.4f})"
        )
        return checkpoint["epoch"] + 1

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
            f"Training {num_epochs} epochs (from {start_epoch}), device={self.device}, AMP={self.use_amp}"
        )

        try:
            for epoch in range(start_epoch, num_epochs + 1):
                self._current_epoch = epoch
                if self._interrupted:
                    self._save_emergency_checkpoint()
                    break

                train_loss = self.train_epoch(train_loader, epoch)
                if self._interrupted:
                    self._save_emergency_checkpoint()
                    break

                val_metrics = self.evaluate(val_loader)
                current_lr = self.optimizer.param_groups[0]["lr"]

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_metrics["val_loss"])
                self.history["val_accuracy"].append(val_metrics["val_accuracy"])
                self.history["val_f1"].append(val_metrics["val_f1"])
                self.history["lr"].append(current_lr)

                logger.info(
                    f"Epoch {epoch}/{num_epochs} - Loss: {train_loss:.4f}, Val F1: {val_metrics['val_f1']:.4f}, LR: {current_lr:.6f}"
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
            self.model.load_state_dict(
                torch.load(best_path, weights_only=False)["model_state_dict"]
            )

        return {
            "best_epoch": best_epoch,
            "best_f1": best_f1,
            "training_time_seconds": training_time,
            "history": self.history,
            "interrupted": self._interrupted,
        }
