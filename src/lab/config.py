"""Typed, serializable configuration for reproducible experiments."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_id: str
    dataset_id: str
    seed: int = 42
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quantization: str = "none"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment name must not be empty")
        if not self.model_id.strip() or not self.dataset_id.strip():
            raise ValueError("model_id and dataset_id are required")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_train_epochs < 1:
            raise ValueError("num_train_epochs must be >= 1")
        if self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.quantization not in {"none", "int8", "int4"}:
            raise ValueError("quantization must be one of: none, int8, int4")

    def to_json(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
