"""Full experiment orchestration: train -> submit -> sync.

Usage:
    uv run python run_experiment.py
    uv run python run_experiment.py --data-dir /path/to/dataset
    uv run python run_experiment.py training.epochs=50 training.lr=0.0003
    uv run python run_experiment.py --skip-submit
    uv run python run_experiment.py training=fast
    uv run python run_experiment.py training.use_compile=true  # Faster training
    uv run python run_experiment.py training.use_class_weights=true  # Imbalanced data
    uv run python run_experiment.py training.resume_from=outputs/run_xyz/latest_checkpoint.pth
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.config import Config
from src.data import create_dataloaders
from src.database import ExperimentDatabase
from src.model import count_parameters, create_model, get_model_size_mb
from src.server import ServerAPI
from src.trainer import Trainer, create_scheduler
from src.utils import get_device, set_seed, setup_logging


@dataclass
class TrainResult:
    best_epoch: int
    best_f1: float
    training_time: float
    model_path: str
    model_size_mb: float
    num_params: int
    device: str


class ExperimentRunner:
    """Orchestrates the full experiment pipeline: train -> submit -> sync."""

    def __init__(self, config: Config, cfg: DictConfig, skip_submit: bool = False) -> None:
        self.config = config
        self.cfg = cfg
        self.skip_submit = skip_submit

        self.run_dir = config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(self.run_dir)
        self.db = ExperimentDatabase("experiments.db")
        self.train_result: TrainResult | None = None

        self._save_config()
        self._create_db_record()

    def _save_config(self) -> None:
        OmegaConf.save(self.cfg, self.run_dir / "config.yaml")

    def _create_db_record(self) -> None:
        db_config = {
            "model_name": self.config.model.name,
            "num_classes": self.config.model.num_classes,
            "dropout": self.config.model.dropout,
            "epochs": self.config.training.epochs,
            "batch_size": self.config.data.batch_size,
            "learning_rate": self.config.training.lr,
            "weight_decay": self.config.training.weight_decay,
            "scheduler": self.config.training.scheduler,
            "data_root": str(self.config.data.root),
            "train_split": self.config.data.train_split,
            "img_size": self.config.data.img_size,
        }
        try:
            self.db.create_experiment(self.config.run_name, db_config)
            self.logger.info(f"Created experiment record: {self.config.run_name}")
        except Exception as e:
            self.logger.warning(f"Experiment may already exist: {e}")

    def train(self) -> TrainResult:
        self.logger.info("=" * 60)
        self.logger.info("Step 1: Training")
        self.logger.info("=" * 60)

        device = get_device()
        self.logger.info(f"Run name: {self.config.run_name}")
        self.logger.info(f"Device: {device}")

        # Set seeds for reproducibility
        set_seed(self.config.data.seed, deterministic=True)
        self.logger.info(f"Set random seed: {self.config.data.seed}")

        # Create dataloaders
        self.logger.info(f"Loading data from {self.config.data.root}")
        train_loader, val_loader, data_info = create_dataloaders(
            data_root=self.config.data.root,
            batch_size=self.config.data.batch_size,
            train_split=self.config.data.train_split,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            seed=self.config.data.seed,
            img_size=self.config.data.img_size,
        )
        self.logger.info(
            f"Train: {data_info['train_size']}, Val: {data_info['val_size']}, "
            f"Classes: {data_info['num_classes']}"
        )

        # Use actual num_classes from dataset (not config) to prevent mismatch bugs
        actual_num_classes = cast(int, data_info["num_classes"])
        if self.config.model.num_classes != actual_num_classes:
            self.logger.warning(
                f"Config specifies {self.config.model.num_classes} classes but dataset has "
                f"{actual_num_classes} classes. Using dataset value."
            )

        model = create_model(
            name=self.config.model.name,
            num_classes=actual_num_classes,
            dropout=self.config.model.dropout,
        ).to(device)

        num_params = count_parameters(model)
        size_mb = get_model_size_mb(model)
        self.logger.info(
            f"Model: {self.config.model.name} ({num_params:,} params, ~{size_mb:.2f} MB)"
        )

        # advanced optimization: torch.compile() :)
        original_model = model
        if self.config.training.use_compile and hasattr(torch, "compile"):
            self.logger.info("Compiling model with torch.compile() for faster training...")
            model = torch.compile(model)  # type: ignore[assignment]
        elif self.config.training.use_compile:
            self.logger.warning(
                "torch.compile() requested but not available (requires PyTorch 2.0+)"
            )

        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        scheduler = create_scheduler(
            optimizer, self.config.training.scheduler, self.config.training.epochs
        )

        # Get class weights if configured
        class_weights = (
            data_info["class_weights"] if self.config.training.use_class_weights else None
        )
        if class_weights is not None:
            class_weights = cast(torch.Tensor, class_weights)

        # Train
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            run_dir=self.run_dir,
            use_amp=self.config.training.use_amp,
            grad_clip=self.config.training.grad_clip,
            early_stopping_patience=self.config.training.early_stopping_patience,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            class_weights=class_weights,
        )

        # Resume from checkpoint if specified
        start_epoch = 1
        if self.config.training.resume_from:
            resume_path = Path(self.config.training.resume_from)
            if resume_path.exists():
                start_epoch = trainer.load_checkpoint(resume_path)
            else:
                self.logger.warning(f"Checkpoint not found: {resume_path}, starting from scratch")

        result = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config.training.epochs,
            start_epoch=start_epoch,
        )

        # Save TorchScript using ORIGINAL (non-compiled) model
        model_path = self.run_dir / "model.pt"
        original_model.eval()
        original_model.cpu()

        # Load best weights into original model
        best_path = self.run_dir / "best_model.pth"
        if best_path.exists():
            best_checkpoint = torch.load(best_path, weights_only=False)
            original_model.load_state_dict(best_checkpoint["model_state_dict"])

        scripted = torch.jit.script(original_model)
        scripted.save(str(model_path))  # type: ignore[no-untyped-call]
        final_size = model_path.stat().st_size / (1024**2)
        self.logger.info(f"Saved TorchScript model to {model_path} ({final_size:.2f} MB)")

        self.train_result = TrainResult(
            best_epoch=result["best_epoch"],
            best_f1=result["best_f1"],
            training_time=result["training_time_seconds"],
            model_path=str(model_path),
            model_size_mb=final_size,
            num_params=num_params,
            device=str(device),
        )

        # Update DB
        self.db.update_experiment(
            self.config.run_name,
            best_epoch=self.train_result.best_epoch,
            best_val_f1=self.train_result.best_f1,
            training_duration_seconds=self.train_result.training_time,
            model_path=self.train_result.model_path,
            model_size_mb=self.train_result.model_size_mb,
            num_parameters=self.train_result.num_params,
            device=self.train_result.device,
        )

        self.logger.info(f"Training complete! Best F1: {self.train_result.best_f1:.4f}")
        if result.get("interrupted"):
            self.logger.info("Note: Training was interrupted. Resume with:")
            self.logger.info(f"  training.resume_from={self.run_dir / 'latest_checkpoint.pth'}")
        return self.train_result

    def submit(self) -> bool:
        if self.train_result is None:
            self.logger.error("Must run train() before submit()")
            return False

        server_token = self.cfg.get("server", {}).get("token", "YOUR_TOKEN_HERE")
        server_username = self.cfg.get("server", {}).get("username", "YOUR_USERNAME")
        server_url = self.cfg.get("server", {}).get("url", "http://hadi.cs.virginia.edu:8000")
        leaderboard_path = self.cfg.get("server", {}).get("leaderboard_path", "/leaderboard3")
        timeout = self.cfg.get("server", {}).get("timeout", 1800)

        if server_token == "YOUR_TOKEN_HERE":
            self.logger.warning("No server token configured. Skipping submission.")
            self.logger.info("Configure server.token in configs/config.yaml to enable submission.")
            return False

        self.logger.info("=" * 60)
        self.logger.info("Step 2: Submitting to server")
        self.logger.info("=" * 60)

        server = ServerAPI(server_token, server_username, server_url)

        result = server.submit_model(
            self.train_result.model_path,
            max_retries=3,
            wait_on_rate_limit=True,
            rate_limit_wait_minutes=16,
            max_rate_limit_retries=5,
        )

        if not result or not result.get("success"):
            error = result.get("error") if result else "Unknown error"
            self.logger.error(f"Submission failed: {error}")
            self.db.update_experiment(self.config.run_name, server_status="failed")
            return False

        self.logger.info("Submission successful!")
        self.db.update_experiment(
            self.config.run_name,
            server_submission_id=result.get("attempt"),
            server_status="pending",
        )

        # Wait for evaluation
        self.logger.info("=" * 60)
        self.logger.info("Step 3: Waiting for evaluation")
        self.logger.info("=" * 60)

        if not server.wait_for_evaluation(timeout=timeout, check_interval=30):
            self.logger.error("Evaluation timed out or failed")
            return False

        self.logger.info("Evaluation complete!")

        # Sync metrics from leaderboard
        time.sleep(5)
        metrics = server.get_metrics_from_leaderboard(leaderboard_path)

        if not metrics:
            self.logger.error("Failed to fetch metrics from leaderboard")
            return False

        self.logger.info(f"Rank: #{metrics['server_rank']}")
        self.logger.info(f"F1 Score: {metrics['server_f1_score']:.4f}")
        self.logger.info(f"Model Size: {metrics['server_model_size_mb']:.2f} MB")

        self.db.update_experiment(self.config.run_name, server_status="successful", **metrics)
        self.logger.info("Database updated with server metrics")

        return True

    def run(self) -> None:
        start_time = time.time()

        try:
            self.train()

            if not self.skip_submit:
                self.submit()
            else:
                self.logger.info("Skipping server submission (--skip-submit)")

            self._log_summary(start_time)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            sys.exit(1)

        except Exception as e:
            self.logger.error(f"Error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        finally:
            self.db.close()

    def _log_summary(self, start_time: float) -> None:
        if self.train_result is None:
            return

        elapsed = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT COMPLETE!")
        self.logger.info("=" * 60)
        self.logger.info(f"Run name: {self.config.run_name}")
        self.logger.info(f"Model: {self.train_result.model_path}")
        self.logger.info(f"Best F1: {self.train_result.best_f1:.4f}")
        self.logger.info(f"Total time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
        self.logger.info("=" * 60)


def parse_custom_args() -> bool:
    """Parse custom CLI args and convert to Hydra overrides. Returns skip_submit flag."""
    skip_submit = False
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--skip-submit":
            skip_submit = True
            i += 1
        elif arg == "--data-dir" and i + 1 < len(sys.argv):
            new_argv.append(f"data.root={sys.argv[i + 1]}")
            i += 2
        elif arg.startswith("--data-dir="):
            path = arg.split("=", 1)[1]
            new_argv.append(f"data.root={path}")
            i += 1
        else:
            new_argv.append(arg)
            i += 1
    sys.argv = new_argv
    return skip_submit


_SKIP_SUBMIT = False


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    config = Config(**cfg_dict)

    runner = ExperimentRunner(config, cfg, skip_submit=_SKIP_SUBMIT)
    runner.run()


if __name__ == "__main__":
    _SKIP_SUBMIT = parse_custom_args()
    main()
