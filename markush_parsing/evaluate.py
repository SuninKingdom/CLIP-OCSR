import json
import logging
import os
import re
from collections import Counter

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from stable_parser import compute_stable_scores

logger = logging.getLogger(__name__)

# Standard chemistry bracket content to exclude when extracting R-group variables
EXCLUDE_BRACKET_ITEMS = ['C@H', 'C@', 'C@@H', 'C@@', 'N+', 'O-', 'C-', 'CH3']
POSITION_VARIABLE_PATTERN = re.compile(r'\[[^\[\]]*\$\]')


def _extract_variables(smiles: str, exclude_items: list[str] | None = None) -> list[str]:
    """Extract R-group variable labels from bracket notation in SMILES.

    e.g. "[R1]C1=CC=C([R2])C=C1" -> ["R1", "R2"]
    Filters out standard chemistry notations like C@H, N+, etc.
    """
    if exclude_items is None:
        exclude_items = EXCLUDE_BRACKET_ITEMS
    contents = re.findall(r'\[(.*?)\]', smiles)
    return [c for c in contents if c not in exclude_items]


def _compare_substitution_or_frequency(
    gt_smiles: str,
    pred_smiles: str | None,
    exclude_items: list[str],
) -> bool:
    """Compare substitution/frequency variations using compare_subvar3 logic.

    Logic (from compare_subvar3):
    1. Extract R-group variables from both SMILES
    2. Check variable multisets match
    3. Replace each unique variable with a dummy atom (P(C), P(CC), ...)
    4. Standardize and compare canonical SMILES
    """
    if pred_smiles is None:
        return False

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

    mol_gt = rdMolStandardize.Cleanup(mol_gt)
    mol_pred = rdMolStandardize.Cleanup(mol_pred)

    return Chem.MolToSmiles(mol_gt) == Chem.MolToSmiles(mol_pred)


def _normalize_pseudo_smiles_all(value) -> list[str]:
    """Normalize a saved pseudo-SMILES candidate set."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return []


def markush_graphical_evaluation(
    gt_smiles: str,
    pred_smiles: str | None,
    pseudo_smiles_all=None,
    exclude_items: list[str] | None = None,
) -> dict:
    """Evaluate a Markush graph and return an auditable decision.

    Substitution and frequency variations use the original graph comparison.
    For position variations, the prediction must contain the same ``[label$]``
    multiset and, after removing ``$``, match a structure in
    ``pseudo_smiles_all``.
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
    result = {
        "markush_graphical_accuracy": False,
        "graphical_evaluation_method": method,
        "position_variation_candidate_count": len(candidates),
        "matched_pseudo_smiles_index": None,
    }

    if pred_smiles is None:
        result["graphical_evaluation_reason"] = "missing_prediction"
        return result

    if gt_smiles == pred_smiles:
        result.update({
            "markush_graphical_accuracy": True,
            "graphical_evaluation_reason": "exact_pseudo_smiles_match",
        })
        return result

    if gt_position_variables and candidates:
        pred_position_variables = POSITION_VARIABLE_PATTERN.findall(pred_smiles)
        if Counter(gt_position_variables) != Counter(pred_position_variables):
            result.update({
                "graphical_evaluation_reason": "position_variable_mismatch",
                "gt_position_variables": gt_position_variables,
                "pred_position_variables": pred_position_variables,
            })
            return result

        pred_without_dollar = pred_smiles.replace("$", "")
        for index, candidate in enumerate(candidates):
            if _compare_substitution_or_frequency(
                candidate, pred_without_dollar, exclude_items
            ):
                result.update({
                    "markush_graphical_accuracy": True,
                    "graphical_evaluation_reason": "matched_pseudo_smiles_all",
                    "matched_pseudo_smiles_index": index,
                })
                return result
        result["graphical_evaluation_reason"] = "no_candidate_match"
        return result

    matched = _compare_substitution_or_frequency(
        gt_smiles, pred_smiles, exclude_items
    )
    result.update({
        "markush_graphical_accuracy": matched,
        "graphical_evaluation_reason": (
            "canonical_graph_match" if matched else "canonical_graph_mismatch"
        ),
    })
    return result


def markush_graphical_accuracy(
    gt_smiles: str,
    pred_smiles: str | None,
    exclude_items: list[str] | None = None,
    pseudo_smiles_all=None,
) -> bool:
    """Backward-compatible boolean wrapper around the detailed evaluator."""
    return markush_graphical_evaluation(
        gt_smiles,
        pred_smiles,
        pseudo_smiles_all=pseudo_smiles_all,
        exclude_items=exclude_items,
    )["markush_graphical_accuracy"]


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
