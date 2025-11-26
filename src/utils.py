import logging
import random
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from .schema import GRAPHABLE_METRICS, EpochHistory


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(run_dir: Path, name: str = "training") -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(run_dir / "training.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(ch)

    return logger


def plot_training_curves(history: EpochHistory, save_dir: Path) -> None:
    """Plot training curves for metrics in GRAPHABLE_METRICS."""
    hist = cast(dict[str, list[float]], history)

    available_metrics = [m for m in GRAPHABLE_METRICS if m in hist and hist[m]]

    if not available_metrics:
        return

    first_metric = available_metrics[0]
    epochs = range(1, len(hist[first_metric]) + 1)

    loss_metrics = [m for m in available_metrics if "loss" in m]
    other_metrics = [m for m in available_metrics if "loss" not in m]

    num_plots = (1 if loss_metrics else 0) + len(other_metrics)
    if num_plots == 0:
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0
    colors = ["b", "r", "g", "orange", "purple"]

    if loss_metrics:
        for i, metric in enumerate(loss_metrics):
            label = metric.replace("_", " ").title()
            axes[plot_idx].plot(epochs, hist[metric], f"{colors[i % len(colors)]}-", label=label)
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel("Loss")
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

    for metric in other_metrics:
        label = metric.replace("_", " ").title()
        axes[plot_idx].plot(epochs, hist[metric], "g-", label=label)
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel(label)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close(fig)
