import torch
import torch.nn as nn


class DemoCNN(nn.Module):
    """CNN demo"""

    def __init__(self, num_classes: int = 10, **kwargs: object) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


def create_model(name: str, num_classes: int = 10, **kwargs: object) -> nn.Module:
    if name == "demo_cnn":
        return DemoCNN(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model: {name}. Available: demo_cnn")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    return count_parameters(model) * 4 / (1024**2)
