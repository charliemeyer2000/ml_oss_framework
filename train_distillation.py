"""Knowledge distillation training.

Usage:
    # With a local teacher checkpoint:
    uv run python train_distillation.py teacher.checkpoint=path/to/teacher.pt

    # With a HuggingFace model (requires HF token):
    uv run python train_distillation.py teacher.hf_model=google/medsiglip-448 hf_token=YOUR_TOKEN
"""

import os
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModel

from src.config import Config
from src.data import create_dataloaders
from src.distillation import DistillationTrainer
from src.model import count_parameters, create_model, get_model_size_mb
from src.trainer import create_scheduler
from src.utils import get_device, plot_training_curves, set_seed, setup_logging


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


def setup_hf_auth(token: str | None) -> None:
    """Setup HuggingFace authentication."""
    # Check env var first
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    final_token = token or env_token

    if not final_token:
        print("\n" + "=" * 60)
        print("ERROR: HuggingFace token required for downloading teacher model!")
        print("=" * 60)
        print("\nTo fix this, do ONE of the following:\n")
        print("1. Set hf_token in config:")
        print("   uv run python train_distillation.py hf_token=YOUR_TOKEN ...\n")
        print("2. Set HF_TOKEN environment variable:")
        print("   export HF_TOKEN=YOUR_TOKEN\n")
        print("3. Login via CLI:")
        print("   uv run huggingface-cli login\n")
        print("Get your token at: https://huggingface.co/settings/tokens")
        print("=" * 60 + "\n")
        sys.exit(1)

    try:
        from huggingface_hub import login

        login(token=final_token, add_to_git_credential=False)
        print("✓ HuggingFace authentication successful")
    except Exception as e:
        print(f"\n❌ HuggingFace login failed: {e}")
        print("Check your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)


def load_hf_teacher(model_name: str, device: torch.device, num_classes: int) -> torch.nn.Module:
    """Load MedSigLIP from HuggingFace and add classifier head."""
    print(f"Loading teacher from HuggingFace: {model_name}")
    teacher = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    with torch.no_grad():
        sample = torch.randn(1, 3, 448, 448).to(device)
        hidden_dim = teacher.vision_model(sample).pooler_output.shape[1]
    teacher.classifier_head = torch.nn.Linear(hidden_dim, num_classes).to(device)
    print(f"  Classifier head: {hidden_dim} -> {num_classes}")

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            x = torch.nn.functional.interpolate(
                x, size=(448, 448), mode="bilinear", align_corners=False
            )
            features = self.model.vision_model(x).pooler_output
            return self.model.classifier_head(features)

    return Wrapper(teacher).to(device)


def load_local_teacher(path: str, device: torch.device) -> torch.nn.Module:
    """Load teacher from local checkpoint."""
    p = Path(path)
    if not p.exists():
        print(f"\n❌ Teacher checkpoint not found: {path}")
        sys.exit(1)

    print(f"Loading teacher from: {path}")
    teacher = torch.jit.load(str(p), map_location=device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


@hydra.main(version_base=None, config_path="configs", config_name="config_distillation")
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

    # Load teacher - either from HF or local checkpoint
    hf_model = cfg.get("teacher", {}).get("hf_model")
    local_checkpoint = cfg.get("teacher", {}).get("checkpoint")
    hf_token = cfg.get("hf_token")

    if hf_model:
        setup_hf_auth(hf_token)
        teacher = load_hf_teacher(hf_model, device, num_classes=num_classes)
    elif local_checkpoint:
        teacher = load_local_teacher(local_checkpoint, device)
    else:
        print("\n" + "=" * 60)
        print("ERROR: No teacher model specified!")
        print("=" * 60)
        print("\nSpecify a teacher using ONE of these options:\n")
        print("1. Local TorchScript checkpoint:")
        print("   teacher.checkpoint=path/to/teacher.pt\n")
        print("2. HuggingFace model (requires hf_token):")
        print("   teacher.hf_model=google/medsiglip-448 hf_token=YOUR_TOKEN")
        print("=" * 60 + "\n")
        sys.exit(1)

    teacher = teacher.to(device)
    logger.info(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")

    student = create_model(config.model.name, num_classes=num_classes).to(device)
    logger.info(
        f"Student: {config.model.name}, {count_parameters(student):,} params, {get_model_size_mb(student):.2f} MB"
    )

    assert config.distillation is not None
    logger.info(f"Distillation: T={config.distillation.temperature}, α={config.distillation.alpha}")

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    scheduler = create_scheduler(optimizer, config.training.scheduler, config.training.epochs)

    class_weights = data_info["class_weights"] if config.training.use_class_weights else None
    if class_weights is not None:
        class_weights = cast(torch.Tensor, class_weights)

    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        device=device,
        temperature=config.distillation.temperature,
        alpha=config.distillation.alpha,
        use_amp=config.training.use_amp,
        grad_clip=config.training.grad_clip,
        class_weights=class_weights,
    )

    best_f1, best_epoch = 0.0, 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_accuracy": [],
    }

    for epoch in range(1, config.training.epochs + 1):
        train_m = trainer.train_epoch(train_loader)
        val_m = trainer.validate(val_loader)
        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["val_f1"].append(val_m["f1"])
        history["val_accuracy"].append(val_m["acc"])

        logger.info(
            f"Epoch {epoch}/{config.training.epochs} | "
            f"Loss: {train_m['loss']:.4f} (hard={train_m['hard']:.4f}, soft={train_m['soft']:.4f}) | "
            f"Val F1: {val_m['f1']:.4f}"
        )

        if val_m["f1"] > best_f1:
            best_f1, best_epoch = val_m["f1"], epoch
            torch.save(
                {"epoch": epoch, "model_state_dict": student.state_dict(), "f1": best_f1},
                run_dir / "best_model.pth",
            )
            logger.info(f"  -> Best! F1={best_f1:.4f}")

        if epoch - best_epoch >= config.training.early_stopping_patience:
            logger.info(f"Early stop at epoch {epoch}")
            break

    plot_training_curves(history, run_dir)

    student.eval()
    student.cpu()
    ckpt = torch.load(run_dir / "best_model.pth", weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])

    model_path = run_dir / "model.pt"
    torch.jit.script(student).save(str(model_path))
    size_mb = model_path.stat().st_size / (1024**2)

    logger.info("=" * 50)
    logger.info(f"Done! Best F1: {best_f1:.4f} @ epoch {best_epoch}")
    logger.info(f"Model: {model_path} ({size_mb:.2f} MB)")
    logger.info("=" * 50)


if __name__ == "__main__":
    parse_args()
    main()
