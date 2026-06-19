import json
import logging
import os
import re
from collections import Counter

from rdkit import Chem
from rdkit.Chem import MolStandardize

from stable_parser import compute_stable_scores

logger = logging.getLogger(__name__)

# Standard chemistry bracket content to exclude when extracting R-group variables
EXCLUDE_BRACKET_ITEMS = ['C@H', 'C@', 'C@@H', 'C@@', 'N+', 'O-', 'C-', 'CH3']


def _extract_variables(smiles: str, exclude_items: list[str] | None = None) -> list[str]:
    """Extract R-group variable labels from bracket notation in SMILES.

    e.g. "[R1]C1=CC=C([R2])C=C1" -> ["R1", "R2"]
    Filters out standard chemistry notations like C@H, N+, etc.
    """
    if exclude_items is None:
        exclude_items = EXCLUDE_BRACKET_ITEMS
    contents = re.findall(r'\[(.*?)\]', smiles)
    return [c for c in contents if c not in exclude_items]


def markush_graphical_accuracy(gt_smiles: str, pred_smiles: str | None, exclude_items: list[str] | None = None) -> bool:
    """Check if predicted SMILES has the same molecular graph as GT.

    Logic (from compare_subvar3):
    1. Extract R-group variables from both SMILES
    2. Check variable multisets match
    3. Replace each unique variable with a dummy atom (P(C), P(CC), ...)
    4. Standardize and compare canonical SMILES
    """
    if pred_smiles is None:
        return False

    if exclude_items is None:
        exclude_items = EXCLUDE_BRACKET_ITEMS

    if gt_smiles == pred_smiles:
        return True

    vars_gt = _extract_variables(gt_smiles, exclude_items)
    vars_pred = _extract_variables(pred_smiles, exclude_items)

    if Counter(vars_gt) != Counter(vars_pred):
        return False

    # Replace variable substituents with real, specific groups
    unique_vars = list(set(vars_gt))
    gt_replaced = gt_smiles
    pred_replaced = pred_smiles
    for i, var in enumerate(unique_vars):
        dummy = f"P({ 'C' * (i + 1) })"
        gt_replaced = gt_replaced.replace(f"[{var}]", dummy)
        pred_replaced = pred_replaced.replace(f"[{var}]", dummy)

    mol_gt = Chem.MolFromSmiles(gt_replaced)
    mol_pred = Chem.MolFromSmiles(pred_replaced)
    if mol_gt is None or mol_pred is None:
        return False

    standardizer = MolStandardize.Standardizer()
    mol_gt = standardizer.standardize(mol_gt)
    mol_pred = standardizer.standardize(mol_pred)

    return Chem.MolToSmiles(mol_gt) == Chem.MolToSmiles(mol_pred)


def evaluate_smiles(gt_smiles: str, pred_smiles: str | None) -> dict:
    """Evaluate pseudo-SMILES prediction quality.

    For Markush structures, the model outputs pseudo-SMILES with R-group labels
    (e.g. [R1], [R2]). Standard metrics (validity, InChI, Tanimoto) are not
    meaningful for pseudo-SMILES. We only use markush_graphical_accuracy:
    replace R-groups with real substituent groups, then compare canonical SMILES.
    """
    return {
        "markush_graphical_accuracy": markush_graphical_accuracy(gt_smiles, pred_smiles),
    }


def evaluate_single(sample_result: dict) -> dict:
    """Evaluate a single pipeline result.

    sample_result should have:
    - gt_smiles, predicted_smiles
    - gt_variables, predicted_variables
    """
    smiles_scores = evaluate_smiles(
        sample_result["gt_smiles"], sample_result.get("predicted_smiles")
    )
    stable_scores = compute_stable_scores(
        sample_result.get("gt_variables") or sample_result.get("gt_stable", {}),
        sample_result.get("predicted_variables") or sample_result.get("predicted_stable"),
    )

    return {**smiles_scores, **stable_scores}


def compute_aggregate_metrics(results: list) -> dict:
    """Aggregate metrics across all evaluated samples."""
    total = len(results)
    if total == 0:
        return {}

    errors = sum(1 for r in results if r.get("error"))

    valid_results = [r for r in results if not r.get("error") and r.get("scores")]

    if not valid_results:
        return {"total_samples": total, "errors": errors, "evaluated": 0}

    scores_list = [r["scores"] for r in valid_results]
    n = len(scores_list)

    def mean(key):
        vals = [s.get(key, 0) for s in scores_list]
        return round(sum(vals) / n, 4) if n else 0.0

    def accuracy(key):
        vals = [s.get(key, False) for s in scores_list]
        return round(sum(1 for v in vals if v) / n, 4) if n else 0.0

    return {
        "total_samples": total,
        "errors": errors,
        "evaluated": n,
        "markush_graphical_accuracy": accuracy("markush_graphical_accuracy"),
        "variable_recall_mean": mean("variable_recall"),
        "variable_precision_mean": mean("variable_precision"),
        "variable_f1_mean": mean("variable_f1"),
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
