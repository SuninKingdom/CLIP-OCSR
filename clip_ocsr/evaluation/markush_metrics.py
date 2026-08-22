"""Evaluate Markush backbone recognition from pseudo-SMILES predictions.

This module contains the graphical-structure metric used by the multimodal
Markush parsing workflow, but it has no dependency on MinerU or an LLM.  It can
therefore be used directly to evaluate CLIP-OCSR predictions against reviewed
Markush pseudo-SMILES labels.

Run ``python -m clip_ocsr.evaluation.markush_metrics --help`` for the batch
CSV/JSON command-line interface.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


# Standard chemistry bracket content to exclude when extracting R-group labels.
EXCLUDE_BRACKET_ITEMS = ["C@H", "C@", "C@@H", "C@@", "N+", "O-", "C-", "CH3"]
POSITION_VARIABLE_PATTERN = re.compile(r"\[[^\[\]]*\$\]")

IDENTIFIER_COLUMNS = (
    "image_name",
    "Image_Name",
    "Image_name",
    "Image Filename",
    "filename",
    "file_name",
)
GROUND_TRUTH_COLUMNS = (
    "final_pseudo_smiles",
    "SMILES",
    "gt_smiles",
    "ground_truth_smiles",
)
PREDICTION_COLUMNS = (
    "Predicted_SMILES",
    "predicted_smiles",
    "prediction",
    "pseudo_smiles",
)
CANDIDATE_COLUMNS = (
    "final_pseudo_smiles_all",
    "SMILES_all",
    "pseudo_smiles_all",
)


def _extract_variables(
    smiles: str, exclude_items: Sequence[str] | None = None
) -> list[str]:
    """Extract bracketed Markush variable labels from a pseudo-SMILES."""
    if exclude_items is None:
        exclude_items = EXCLUDE_BRACKET_ITEMS
    contents = re.findall(r"\[(.*?)\]", smiles)
    return [item for item in contents if item not in exclude_items]


def _compare_substitution_or_frequency(
    gt_smiles: str,
    pred_smiles: str | None,
    exclude_items: Sequence[str],
) -> bool:
    """Compare substitution/frequency variants after graph normalization."""
    if pred_smiles is None:
        return False
    if gt_smiles == pred_smiles:
        return True

    variables_gt = _extract_variables(gt_smiles, exclude_items)
    variables_pred = _extract_variables(pred_smiles, exclude_items)
    if Counter(variables_gt) != Counter(variables_pred):
        return False

    gt_replaced = gt_smiles
    pred_replaced = pred_smiles
    for index, variable in enumerate(sorted(set(variables_gt))):
        dummy = f"P({'C' * (index + 1)})"
        gt_replaced = gt_replaced.replace(f"[{variable}]", dummy)
        pred_replaced = pred_replaced.replace(f"[{variable}]", dummy)

    mol_gt = Chem.MolFromSmiles(gt_replaced)
    mol_pred = Chem.MolFromSmiles(pred_replaced)
    if mol_gt is None or mol_pred is None:
        return False

    mol_gt = rdMolStandardize.Cleanup(mol_gt)
    mol_pred = rdMolStandardize.Cleanup(mol_pred)
    return Chem.MolToSmiles(mol_gt) == Chem.MolToSmiles(mol_pred)


def _normalize_pseudo_smiles_all(value: Any) -> list[str]:
    """Normalize a saved pseudo-SMILES candidate set."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        text = str(value).strip()
        return [text] if text else []

    text = value.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(";") if item.strip()]


def markush_graphical_evaluation(
    gt_smiles: str,
    pred_smiles: str | None,
    pseudo_smiles_all: Any = None,
    exclude_items: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate one predicted Markush pseudo-SMILES with an audit decision.

    Substitution and frequency variations are compared after replacing Markush
    variables with deterministic dummy fragments and canonicalizing the RDKit
    molecular graphs.  For a positional variable marked with ``$``, a correct
    prediction must retain the same positional-variable labels and match one of
    the reviewed structures supplied in ``pseudo_smiles_all`` after ``$`` is
    removed.
    """
    if exclude_items is None:
        exclude_items = EXCLUDE_BRACKET_ITEMS

    candidates = _normalize_pseudo_smiles_all(pseudo_smiles_all)
    gt_position_variables = POSITION_VARIABLE_PATTERN.findall(gt_smiles or "")
    method = (
        "position_variation_smiles_set"
        if gt_position_variables and candidates
        else "position_variation_strict_fallback"
        if gt_position_variables
        else "substitution_or_frequency_graph"
    )
    result: dict[str, Any] = {
        "markush_graphical_accuracy": False,
        "graphical_evaluation_method": method,
        "position_variation_candidate_count": len(candidates),
        "matched_pseudo_smiles_index": None,
    }

    if pred_smiles is None or not str(pred_smiles).strip():
        result["graphical_evaluation_reason"] = "missing_prediction"
        return result
    pred_smiles = str(pred_smiles).strip()

    if gt_smiles == pred_smiles:
        result.update(
            {
                "markush_graphical_accuracy": True,
                "graphical_evaluation_reason": "exact_pseudo_smiles_match",
            }
        )
        return result

    if gt_position_variables and candidates:
        pred_position_variables = POSITION_VARIABLE_PATTERN.findall(pred_smiles)
        if Counter(gt_position_variables) != Counter(pred_position_variables):
            result.update(
                {
                    "graphical_evaluation_reason": "position_variable_mismatch",
                    "gt_position_variables": gt_position_variables,
                    "pred_position_variables": pred_position_variables,
                }
            )
            return result

        pred_without_dollar = pred_smiles.replace("$", "")
        for index, candidate in enumerate(candidates):
            if _compare_substitution_or_frequency(
                candidate, pred_without_dollar, exclude_items
            ):
                result.update(
                    {
                        "markush_graphical_accuracy": True,
                        "graphical_evaluation_reason": "matched_pseudo_smiles_all",
                        "matched_pseudo_smiles_index": index,
                    }
                )
                return result
        result["graphical_evaluation_reason"] = "no_candidate_match"
        return result

    matched = _compare_substitution_or_frequency(
        gt_smiles, pred_smiles, exclude_items
    )
    result.update(
        {
            "markush_graphical_accuracy": matched,
            "graphical_evaluation_reason": (
                "canonical_graph_match" if matched else "canonical_graph_mismatch"
            ),
        }
    )
    return result


def markush_graphical_accuracy(
    gt_smiles: str,
    pred_smiles: str | None,
    exclude_items: Sequence[str] | None = None,
    pseudo_smiles_all: Any = None,
) -> bool:
    """Return only the boolean Markush graphical-accuracy decision."""
    return bool(
        markush_graphical_evaluation(
            gt_smiles,
            pred_smiles,
            pseudo_smiles_all=pseudo_smiles_all,
            exclude_items=exclude_items,
        )["markush_graphical_accuracy"]
    )


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        records = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Expected an object at {input_path}:{line_number}"
                    )
                records.append(value)
        return records
    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict) and isinstance(value.get("results"), list):
            value = value["results"]
        if not isinstance(value, list) or not all(
            isinstance(record, dict) for record in value
        ):
            raise ValueError(f"Expected a JSON list of objects in {input_path}")
        return [dict(record) for record in value]
    raise ValueError(
        f"Unsupported format {suffix!r} for {input_path}; use CSV, JSON, or JSONL"
    )


def _resolve_column(
    records: Sequence[Mapping[str, Any]],
    requested: str | None,
    aliases: Sequence[str],
    description: str,
    *,
    required: bool = True,
) -> str | None:
    columns = {str(key) for record in records for key in record}
    if requested:
        if requested not in columns:
            raise ValueError(
                f"{description} column {requested!r} was not found; "
                f"available columns: {sorted(columns)}"
            )
        return requested
    for alias in aliases:
        if alias in columns:
            return alias
    if required:
        raise ValueError(
            f"Could not detect the {description} column; available columns: "
            f"{sorted(columns)}"
        )
    return None


def _sample_identifier(value: Any) -> str:
    return Path(str(value).strip().replace("\\", "/")).name


def _index_records(
    records: Sequence[Mapping[str, Any]], identifier_column: str, source: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row_number, record in enumerate(records, start=2):
        identifier = _sample_identifier(record.get(identifier_column, ""))
        if not identifier:
            raise ValueError(f"Missing sample identifier in {source} row {row_number}")
        if identifier in indexed:
            raise ValueError(f"Duplicate sample identifier {identifier!r} in {source}")
        indexed[identifier] = record
    return indexed


def evaluate_prediction_files(
    labels_path: str | Path,
    predictions_path: str | Path,
    *,
    label_id_column: str | None = None,
    prediction_id_column: str | None = None,
    label_smiles_column: str | None = None,
    prediction_smiles_column: str | None = None,
    candidate_column: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate prediction records against labels, retaining every label row."""
    labels = _load_records(labels_path)
    predictions = _load_records(predictions_path)
    if not labels:
        raise ValueError(f"No label records found in {labels_path}")

    label_id_column = _resolve_column(
        labels, label_id_column, IDENTIFIER_COLUMNS, "label identifier"
    )
    prediction_id_column = _resolve_column(
        predictions,
        prediction_id_column,
        IDENTIFIER_COLUMNS,
        "prediction identifier",
    )
    label_smiles_column = _resolve_column(
        labels, label_smiles_column, GROUND_TRUTH_COLUMNS, "ground-truth SMILES"
    )
    prediction_smiles_column = _resolve_column(
        predictions,
        prediction_smiles_column,
        PREDICTION_COLUMNS,
        "predicted SMILES",
    )
    candidate_column = _resolve_column(
        labels,
        candidate_column,
        CANDIDATE_COLUMNS,
        "position-variation candidate",
        required=False,
    )

    label_index = _index_records(labels, label_id_column, str(labels_path))
    prediction_index = _index_records(
        predictions, prediction_id_column, str(predictions_path)
    )

    details: list[dict[str, Any]] = []
    correct = 0
    missing_predictions = 0
    method_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for identifier, label in label_index.items():
        prediction = prediction_index.get(identifier)
        predicted_smiles = (
            prediction.get(prediction_smiles_column) if prediction is not None else None
        )
        if predicted_smiles is not None and not str(predicted_smiles).strip():
            predicted_smiles = None
        if predicted_smiles is None:
            missing_predictions += 1

        gt_smiles = str(label.get(label_smiles_column) or "").strip()
        if not gt_smiles:
            raise ValueError(f"Missing ground-truth SMILES for {identifier!r}")
        candidates = label.get(candidate_column) if candidate_column else None
        decision = markush_graphical_evaluation(
            gt_smiles,
            predicted_smiles,
            pseudo_smiles_all=candidates,
        )
        is_correct = bool(decision["markush_graphical_accuracy"])
        correct += int(is_correct)
        method_counts[decision["graphical_evaluation_method"]] += 1
        reason_counts[decision["graphical_evaluation_reason"]] += 1
        details.append(
            {
                "image_name": identifier,
                "ground_truth_smiles": gt_smiles,
                "predicted_smiles": predicted_smiles,
                **decision,
            }
        )

    extra_identifiers = sorted(set(prediction_index) - set(label_index))
    total = len(label_index)
    summary = {
        "labels_path": str(Path(labels_path)),
        "predictions_path": str(Path(predictions_path)),
        "total_samples": total,
        "correct_samples": correct,
        "incorrect_samples": total - correct,
        "missing_predictions": missing_predictions,
        "extra_predictions": len(extra_identifiers),
        "extra_prediction_identifiers": extra_identifiers,
        "markush_graphical_accuracy": correct / total if total else 0.0,
        "graphical_evaluation_methods": dict(method_counts),
        "graphical_evaluation_reasons": dict(reason_counts),
        "resolved_columns": {
            "label_identifier": label_id_column,
            "prediction_identifier": prediction_id_column,
            "ground_truth_smiles": label_smiles_column,
            "predicted_smiles": prediction_smiles_column,
            "pseudo_smiles_all": candidate_column,
        },
    }
    return summary, details


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Markush pseudo-SMILES predictions using the graphical "
            "accuracy metric used by CLIP-OCSR."
        )
    )
    parser.add_argument("--labels", required=True, help="Label CSV, JSON, or JSONL")
    parser.add_argument(
        "--predictions", required=True, help="Prediction CSV, JSON, or JSONL"
    )
    parser.add_argument("--output", help="Optional aggregate metrics JSON path")
    parser.add_argument("--details", help="Optional per-sample audit JSONL path")
    parser.add_argument("--label-id-column")
    parser.add_argument("--prediction-id-column")
    parser.add_argument("--label-smiles-column")
    parser.add_argument("--prediction-smiles-column")
    parser.add_argument("--candidate-column")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        summary, details = evaluate_prediction_files(
            args.labels,
            args.predictions,
            label_id_column=args.label_id_column,
            prediction_id_column=args.prediction_id_column,
            label_smiles_column=args.label_smiles_column,
            prediction_smiles_column=args.prediction_smiles_column,
            candidate_column=args.candidate_column,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if args.output:
        _write_json(args.output, summary)
    if args.details:
        _write_jsonl(args.details, details)

    accuracy = 100.0 * summary["markush_graphical_accuracy"]
    print(f"Total samples: {summary['total_samples']}")
    print(f"Correct samples: {summary['correct_samples']}")
    print(f"Missing predictions: {summary['missing_predictions']}")
    print(f"Markush Graphical Accuracy: {accuracy:.2f}%")
    if args.output:
        print(f"Aggregate metrics: {args.output}")
    if args.details:
        print(f"Per-sample audit: {args.details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
