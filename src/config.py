from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    name: str = "tiny_cnn"
    num_classes: int = Field(default=10, ge=2)
    dropout: float = Field(default=0.3, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    epochs: int = Field(default=20, ge=1)
    lr: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    scheduler: Literal["cosine", "step"] = "cosine"
    early_stopping_patience: int = Field(default=5, ge=1)
    grad_clip: float = Field(default=1.0, gt=0)
    use_amp: bool = True
    use_compile: bool = False
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    use_class_weights: bool = False
    resume_from: str | None = None


class DataConfig(BaseModel):
    root: Path = Path("data/training_dataset")
    train_split: float = Field(default=0.9, gt=0.0, lt=1.0)
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    seed: int = 42
    img_size: int = Field(default=224, ge=32)

    @field_validator("root", mode="before")
    @classmethod
    def convert_root(cls, v: str | Path) -> Path:
        return Path(v)


class PathsConfig(BaseModel):
    output_dir: Path = Path("outputs")

    @field_validator("output_dir", mode="before")
    @classmethod
    def convert_output_dir(cls, v: str | Path) -> Path:
        return Path(v)


class DistillationConfig(BaseModel):
    temperature: float = Field(default=4.0, gt=0)
    alpha: float = Field(default=0.3, ge=0.0, le=1.0)


class TeacherConfig(BaseModel):
    checkpoint: str | None = None
    torchscript: bool = False


class Config(BaseModel):
    run_name: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    distillation: DistillationConfig | None = None
    teacher: TeacherConfig | None = None

    @property
    def run_dir(self) -> Path:
        return self.paths.output_dir / self.run_name
