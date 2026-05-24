"""Stage 2 configuration: OCSR fine-tuning."""

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Stage2Config:
    """Configuration for Stage 2 OCSR fine-tuning."""

    # Data
    dataframe_path: str = "sample_data/stage2/train.csv"
    img_folder: str = "sample_data/stage2"
    tokenizer_path: str = "assets/tokenizer_smiles.json"
    abbrev_group_path: str = "assets/abbrev_group.json"

    # Checkpoint
    preload: int = 0  # Epoch to resume from, 0 = start from scratch
    model_folder: str = "checkpoints/stage2"
    model_basename: str = "stage2_clip_ocsr_"
    tb_folder: str = "runs/stage2"

    # Training
    training_ratio: float = 0.9992
    num_epochs: int = 60
    batch_size: int = 112  # Per GPU
    lr: float = 1.5e-4

    # CLIP encoder
    clip_ckpt_path: str = ""  # Path to Stage 1 checkpoint
    trainable: bool = True

    # Transformer decoder
    seq_len: int = 256
    tgt_seq_len: int = 256
    d_model: int = 512
    N: int = 6
    h: int = 8
    dropout: float = 0.1
    d_ff: int = 2048

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Stage2Config":
        """Load config from a YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


def get_weights_file_path(model_folder, model_basename, epoch: str):
    """Get the file path for a specific epoch's weights."""
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(model_folder, model_basename):
    """Find the latest weights file in the weights folder."""
    model_filename = f"{model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
