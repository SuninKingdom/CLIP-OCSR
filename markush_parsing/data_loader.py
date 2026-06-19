import json
import os
from dataclasses import dataclass, field

from config import Config


@dataclass
class SampleData:
    id: int
    image_name: str
    image_path: str
    gt_smiles: str
    variables_gt: dict = field(default_factory=dict)


def load_dataset(config: Config) -> list:
    """Load all samples from labels JSON file.

    Expected format:
    [{"image_name": "xxx.png", "gt_smiles": "...", "variables": {"R1": [...], ...}}, ...]
    """
    with open(config.labels_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for i, entry in enumerate(raw_data):
        sample = SampleData(
            id=i,
            image_name=entry["image_name"],
            image_path=os.path.join(config.dataset_dir, entry["image_name"]),
            gt_smiles=entry.get("gt_smiles", ""),
            variables_gt=entry.get("variables", {}),
        )
        samples.append(sample)

    return samples
