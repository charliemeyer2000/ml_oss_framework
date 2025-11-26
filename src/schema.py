"""Centralized schema for metrics.

When adding a new epoch metric, update EPOCH_METRICS, EpochHistory, and record_epoch().
Runtime validation catches if EPOCH_METRICS and EpochHistory drift.
"""

from typing import TypedDict

EPOCH_METRICS = ["train_loss", "val_loss", "val_f1", "val_accuracy", "lr"]
GRAPHABLE_METRICS = ["train_loss", "val_loss", "val_f1"]

ALLOWED_SORT_METRICS = frozenset(
    {
        "best_val_f1",
        "best_val_accuracy",
        "best_val_loss",
        "training_duration_seconds",
        "model_size_mb",
        "num_parameters",
        "server_f1_score",
        "server_model_size_mb",
        "server_rank",
    }
)


class EpochHistory(TypedDict):
    train_loss: list[float]
    val_loss: list[float]
    val_f1: list[float]
    val_accuracy: list[float]
    lr: list[float]


def _validate_schema() -> None:
    metrics = set(EPOCH_METRICS)
    typed_keys = set(EpochHistory.__annotations__.keys())
    if metrics != typed_keys:
        raise RuntimeError(
            f"EPOCH_METRICS and EpochHistory out of sync!\n"
            f"  In EPOCH_METRICS only: {metrics - typed_keys}\n"
            f"  In EpochHistory only: {typed_keys - metrics}"
        )


_validate_schema()


def create_empty_history() -> EpochHistory:
    return EpochHistory(
        train_loss=[],
        val_loss=[],
        val_f1=[],
        val_accuracy=[],
        lr=[],
    )


def record_epoch(
    history: EpochHistory,
    *,
    train_loss: float,
    val_loss: float,
    val_f1: float,
    val_accuracy: float,
    lr: float,
) -> None:
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_f1)
    history["val_accuracy"].append(val_accuracy)
    history["lr"].append(lr)
