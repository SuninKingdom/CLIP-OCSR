"""Stage 1 configuration: CLIP domain-adaptive pretraining."""

from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class Stage1Config:
    """Configuration for Stage 1 CLIP domain-adaptive pretraining."""

    # Data
    csv_path: str = "sample_data/stage1/train.csv"
    data_dir: str = "sample_data/stage1"
    checkpoint_dir: str = "checkpoints/stage1"
    tensorboard_dir: str = "runs/stage1"

    # Model
    model_name: str = "RN50"
    pretrained_path: str = ""  # Path to OpenAI CLIP RN50 .pt weights

    # Training
    num_epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 2.0e-6
    weight_decay: float = 1.0e-4
    scheduler_T_max: int = 60
    train_split_ratio: float = 0.995
    num_workers: int = 4
    save_every_epoch: int = 1
    seed: int = 42

    # Derived
    model_basename: str = "stage1_clip_rn50"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Stage1Config":
        """Load config from a YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
