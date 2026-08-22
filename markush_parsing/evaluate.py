import json
import logging
import os
from collections import Counter

from clip_ocsr.evaluation.markush_metrics import (
    markush_graphical_accuracy,
    markush_graphical_evaluation,
)
from stable_parser import compute_stable_scores

logger = logging.getLogger(__name__)

# Re-export the graphical metric for compatibility with existing callers.
__all__ = [
    "markush_graphical_accuracy",
    "markush_graphical_evaluation",
    "evaluate_smiles",
    "evaluate_single",
    "compute_aggregate_metrics",
    "save_results",
]


def evaluate_smiles(
    gt_smiles: str, pred_smiles: str | None, pseudo_smiles_all=None
) -> dict:
    """Evaluate pseudo-SMILES prediction quality.

    For Markush structures, the model outputs pseudo-SMILES with R-group labels
    (e.g. [R1], [R2]). Standard metrics (validity, InChI, Tanimoto) are not
    meaningful for pseudo-SMILES. We only use markush_graphical_accuracy:
    replace R-groups with real substituent groups, then compare canonical SMILES.
    """
    return markush_graphical_evaluation(
        gt_smiles, pred_smiles, pseudo_smiles_all=pseudo_smiles_all
    )


def evaluate_single(sample_result: dict) -> dict:
    """Evaluate a single pipeline result.

    sample_result should have:
    - gt_smiles, predicted_smiles
    - gt_variables, predicted_variables
    """
    pseudo_smiles_all = (
        sample_result.get("gt_pseudo_smiles_all")
        or sample_result.get("pseudo_smiles_all")
        or sample_result.get("final_pseudo_smiles_all")
    )
    smiles_scores = evaluate_smiles(
        sample_result["gt_smiles"],
        sample_result.get("predicted_smiles"),
        pseudo_smiles_all=pseudo_smiles_all,
    )
    stable_scores = compute_stable_scores(
        sample_result.get("gt_variables") or sample_result.get("gt_stable", {}),
        sample_result.get("predicted_variables") or sample_result.get("predicted_stable"),
    )

    return {
        **smiles_scores,
        **stable_scores,
    }


def compute_aggregate_metrics(results: list) -> dict:
    """Aggregate metrics across all evaluated samples."""
    total = len(results)
    if total == 0:
        return {}

    errors = sum(1 for r in results if r.get("error"))

    # End-to-end failures are system failures. Keep every sample in the
    # denominator and assign zero to missing scores.
    scores_list = [
        (r.get("scores") or {}) if not r.get("error") else {}
        for r in results
    ]
    scored = sum(
        1 for r in results if not r.get("error") and r.get("scores")
    )
    n = total

    def raw_mean(key):
        vals = [float(s.get(key, 0.0)) for s in scores_list]
        return sum(vals) / n if n else 0.0

    def accuracy(key):
        vals = [s.get(key, False) for s in scores_list]
        return round(sum(1 for v in vals if v) / n, 4) if n else 0.0

    variable_recall_raw = raw_mean("variable_recall")
    variable_precision_raw = raw_mean("variable_precision")
    variable_f1 = (
        2 * variable_precision_raw * variable_recall_raw
        / (variable_precision_raw + variable_recall_raw)
        if variable_precision_raw + variable_recall_raw
        else 0.0
    )

    return {
        "total_samples": total,
        "errors": errors,
        "evaluated": n,
        "scored_samples": scored,
        "failed_or_unscored_samples": total - scored,
        "graphical_evaluation_methods": dict(Counter(
            score.get("graphical_evaluation_method", "unscored")
            for score in scores_list
        )),
        "markush_graphical_accuracy": accuracy("markush_graphical_accuracy"),
        "variable_recall_mean": round(variable_recall_raw, 4),
        "variable_precision_mean": round(variable_precision_raw, 4),
        # MarkushGrapher-2 headline F1: harmonic mean of dataset macro P/R.
        "variable_f1": round(variable_f1, 4),
        "variable_f1_from_macro_precision_recall": round(variable_f1, 4),
    }


def save_results(results: list, metrics: dict, output_dir: str):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-sample results
    per_sample_path = os.path.join(output_dir, "per_sample.jsonl")
    with open(per_sample_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # Aggregate metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_dir}")
    return per_sample_path, metrics_path
