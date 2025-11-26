"""Standard training (no distillation).

Usage:
    uv run python train.py
    uv run python train.py model=shufflenet_x0_5
    uv run python train.py training.epochs=50 training.lr=0.0003
    uv run python train.py --data-dir /path/to/data
"""

import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.config import Config
from src.data import create_dataloaders
from src.model import count_parameters, create_model, get_model_size_mb
from src.trainer import Trainer, create_scheduler
from src.utils import get_device, set_seed, setup_logging


def parse_args() -> None:
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--data-dir" and i + 1 < len(sys.argv):
            new_argv.append(f"data.root={sys.argv[i + 1]}")
            i += 2
        elif arg.startswith("--data-dir="):
            new_argv.append(f"data.root={arg.split('=', 1)[1]}")
            i += 1
        else:
            new_argv.append(arg)
            i += 1
    sys.argv = new_argv


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    config = Config(**cfg_dict)

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir)

    logger.info(f"Run: {config.run_name}")
    set_seed(config.data.seed, deterministic=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    device = get_device()
    logger.info(f"Device: {device}")

    train_loader, val_loader, data_info = create_dataloaders(
        data_root=config.data.root,
        batch_size=config.data.batch_size,
        train_split=config.data.train_split,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.data.seed,
        img_size=config.data.img_size,
    )
    logger.info(
        f"Train: {data_info['train_size']}, Val: {data_info['val_size']}, Classes: {data_info['num_classes']}"
    )

    num_classes = cast(int, data_info["num_classes"])
    model = create_model(config.model.name, num_classes=num_classes).to(device)
    logger.info(
        f"Model: {config.model.name}, {count_parameters(model):,} params, {get_model_size_mb(model):.2f} MB"
    )

    if config.training.use_compile and hasattr(torch, "compile"):
        logger.info("Compiling model...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    scheduler = create_scheduler(optimizer, config.training.scheduler, config.training.epochs)

    class_weights = data_info["class_weights"] if config.training.use_class_weights else None
    if class_weights is not None:
        class_weights = cast(torch.Tensor, class_weights)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        run_dir=run_dir,
        use_amp=config.training.use_amp,
        grad_clip=config.training.grad_clip,
        early_stopping_patience=config.training.early_stopping_patience,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        class_weights=class_weights,
    )

    start_epoch = 1
    if config.training.resume_from:
        resume_path = Path(config.training.resume_from)
        if resume_path.exists():
            start_epoch = trainer.load_checkpoint(resume_path)

    result = trainer.fit(train_loader, val_loader, config.training.epochs, start_epoch)

    # Export TorchScript
    model.eval()
    model.cpu()
    best_path = run_dir / "best_model.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, weights_only=False)["model_state_dict"])

    model_path = run_dir / "model.pt"
    torch.jit.script(model).save(str(model_path))
    size_mb = model_path.stat().st_size / (1024**2)

    logger.info("=" * 50)
    logger.info(f"Done! Best F1: {result['best_f1']:.4f} @ epoch {result['best_epoch']}")
    logger.info(f"Model: {model_path} ({size_mb:.2f} MB)")
    logger.info("=" * 50)


if __name__ == "__main__":
    parse_args()
    main()
