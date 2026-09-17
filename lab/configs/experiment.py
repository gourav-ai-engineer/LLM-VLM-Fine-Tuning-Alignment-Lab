"""Typed, dependency-free experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by reproducible LoRA experiments."""

    seed: int = 42
    model_name: str = "roberta-base"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    learning_rate: float = 2e-4
    batch_size: int = 8
    epochs: int = 3
    output_dir: str = "artifacts/experiments"

    def validate(self) -> None:
        """Raise ValueError when an experiment setting is invalid."""
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation after validation."""
        self.validate()
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Save this configuration as formatted JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate a configuration from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(**payload)
        config.validate()
        return config
