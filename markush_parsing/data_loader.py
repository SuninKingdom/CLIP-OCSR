import csv
import json
import os
import re
from dataclasses import dataclass, field

from config import Config
from stable_parser import parse_stable_string


@dataclass
class SampleData:
    id: int
    image_name: str
    image_path: str
    gt_smiles: str
    variables_gt: dict = field(default_factory=dict)
    pseudo_smiles_all: list[str] = field(default_factory=list)


def parse_pseudo_smiles_all(value) -> list[str]:
    """Normalize a pseudo-SMILES candidate set from JSON or CSV labels."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        return [str(value).strip()] if str(value).strip() else []

    text = value.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split(";") if item.strip()]


def parse_annotation_variables(annotation) -> dict:
    """Extract the MarkushGrapher ``<stable>`` variable dictionary."""
    if not isinstance(annotation, str):
        return {}
    match = re.search(r"<stable>(.*?)</stable>", annotation, re.DOTALL)
    return parse_stable_string(match.group(1)) if match else {}


def _normalize_label_entry(entry: dict) -> dict:
    """Map supported JSON/CSV label schemas to the project schema."""
    image_name = entry.get("image_name") or entry.get("Image Filename") or ""
    gt_smiles = (
        entry.get("gt_smiles")
        or entry.get("final_pseudo_smiles")
        or entry.get("SMILES")
        or ""
    )
    pseudo_smiles_all = parse_pseudo_smiles_all(
        entry.get("pseudo_smiles_all")
        or entry.get("final_pseudo_smiles_all")
        or entry.get("SMILES_all")
    )

    variables = entry.get("variables")
    if variables is None:
        variables = parse_annotation_variables(entry.get("annotation"))
    if isinstance(variables, str):
        try:
            variables = json.loads(variables)
        except json.JSONDecodeError:
            variables = {}
    if not isinstance(variables, dict):
        variables = {}

    return {
        "image_name": image_name,
        "gt_smiles": gt_smiles,
        "variables": variables,
        "pseudo_smiles_all": pseudo_smiles_all,
    }


def load_label_entries(labels_path: str) -> list[dict]:
    """Load labels from the original JSON schema or reviewed M2S CSV."""
    extension = os.path.splitext(labels_path)[1].lower()
    if extension == ".json":
        with open(labels_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            raise ValueError(f"Expected a JSON list in {labels_path}")
    elif extension == ".csv":
        with open(labels_path, "r", encoding="utf-8-sig", newline="") as f:
            raw_data = list(csv.DictReader(f))
    else:
        raise ValueError(
            f"Unsupported labels format {extension!r}; expected .json or .csv"
        )

    entries = [_normalize_label_entry(entry) for entry in raw_data]
    missing_names = [
        index + 2
        for index, entry in enumerate(entries)
        if not entry["image_name"]
    ]
    if missing_names:
        raise ValueError(f"Labels missing image names at rows: {missing_names}")
    return entries


def load_combined_label_entries(
    labels_path: str,
    variable_labels_path: str | None = None,
) -> list[dict]:
    """Load graphical labels and optionally merge a variable-label file.

    For M2S, the reviewed CSV remains the authoritative graphical label source;
    only the variable dictionary is taken from the supplemental JSON file.
    """
    entries = load_label_entries(labels_path)
    if not variable_labels_path:
        return entries

    variable_entries = load_label_entries(variable_labels_path)
    variable_map = {entry["image_name"]: entry for entry in variable_entries}
    for entry in entries:
        variable_entry = variable_map.get(entry["image_name"])
        if variable_entry is not None:
            entry["variables"] = variable_entry.get("variables", {})
    return entries


def load_dataset(config: Config) -> list:
    """Load all samples from a supported JSON or CSV labels file."""
    raw_data = load_label_entries(config.labels_path)

    samples = []
    for i, entry in enumerate(raw_data):
        samples.append(SampleData(
            id=i,
            image_name=entry["image_name"],
            image_path=os.path.join(config.dataset_dir, entry["image_name"]),
            gt_smiles=entry.get("gt_smiles", ""),
            variables_gt=entry.get("variables", {}),
            pseudo_smiles_all=entry.get("pseudo_smiles_all", []),
        ))
    return samples
