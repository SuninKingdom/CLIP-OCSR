import math
import re


def expand_integer_ranges(stable: dict | None) -> dict | None:
    """Expand standalone integer ranges using MarkushGrapher semantics.

    A complete value such as ``"1-3"`` becomes ``["1", "2", "3"]``.
    Chemical descriptions such as ``"C1-C6 alkyl"`` remain unchanged.
    """
    if stable is None:
        return None

    expanded = {}
    for label, values in stable.items():
        expanded_values = []
        for value in values:
            match = (
                re.fullmatch(r"(\d+)-(\d+)", value)
                if isinstance(value, str)
                else None
            )
            if match is None:
                expanded_values.append(value)
                continue
            start, end = (int(item) for item in match.groups())
            expanded_values.extend(
                str(number)
                for number in range(min(start, end), max(start, end) + 1)
            )
        expanded[label] = expanded_values
    return expanded


def parse_stable_string(stable_str: str) -> dict:
    """Parse a stable-format string into a dict.

    Format: "R1:value1<ns>R2:value2<n>value3<ns>..."
    -> {"R1": ["value1"], "R2": ["value2", "value3"]}
    """
    if not stable_str or not stable_str.strip():
        return {}

    stable = {}
    for item in stable_str.split("<ns>"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":", 1)
        if len(parts) < 2:
            continue
        label_raw = parts[0].strip()
        values_str = parts[1].strip()

        values = [v.strip() for v in values_str.split("<n>") if v.strip()]

        for label in label_raw.split(","):
            label = label.strip()
            if label:
                stable[label] = values

    return expand_integer_ranges(stable)


def normalize_stable(stable: dict) -> dict:
    """Normalize a stable dict for comparison.

    Normalizations:
    1. Lowercase all keys and values
    2. Strip "a " prefix and " group" suffix from values
    3. Strip whitespace
    4. Sort values within each key
    """
    normalized = {}
    for key, values in stable.items():
        norm_key = key.lower().strip()
        norm_values = []
        for v in values:
            v = v.lower().strip()
            v = re.sub(r"^a\s+", "", v)
            v = re.sub(r"\s+group$", "", v)
            norm_values.append(v)
        norm_values.sort()
        normalized[norm_key] = norm_values
    return normalized


def compute_stable_scores(
    gt_stable: dict, pred_stable: dict | None, permissive: bool = True, normalize: bool = True
) -> dict:
    """Compute stable recall, precision, and per-sample F1.

    Replicates the logic from MarkushGrapher's get_stable_score().
    """
    scores = {
        "variable_recall": 0.0,
        "variable_precision": 0.0,
        "variable_f1": 0.0,
    }

    if pred_stable is None:
        return scores

    if gt_stable == {}:
        if pred_stable == {}:
            scores["variable_recall"] = 1.0
            scores["variable_precision"] = 1.0
            scores["variable_f1"] = 1.0
        return scores

    # MarkushGrapher expands these ranges in its tokenizer before calling
    # get_stable_score. Direct JSON predictions need the same preprocessing.
    gt = expand_integer_ranges(dict(gt_stable))
    pred = expand_integer_ranges(dict(pred_stable))

    # Normalize: correct filler words
    if normalize:
        new_pred = {}
        for label, pred_subs in pred.items():
            if label not in gt:
                new_pred[label] = pred_subs
                continue
            normalized_gt = [s.replace("a ", "").replace(" group", "") for s in gt[label]]
            new_pred_subs = []
            for ps in pred_subs:
                if ps in gt[label]:
                    new_pred_subs.append(ps)
                    continue
                norm_ps = ps.replace("a ", "").replace(" group", "")
                if norm_ps in normalized_gt:
                    idx = normalized_gt.index(norm_ps)
                    new_pred_subs.append(gt[label][idx])
                else:
                    new_pred_subs.append(ps)
            new_pred[label] = new_pred_subs
        pred = new_pred

    # Permissive: case-insensitive, space-removed
    if permissive:
        gt = {k.lower(): [e.lower().replace(" ", "") for e in v] for k, v in gt.items()}
        pred = {k.lower(): [e.lower().replace(" ", "") for e in v] for k, v in pred.items()}

    # Compute recall
    gt_found = []
    for label, gt_subs in gt.items():
        if label not in pred:
            gt_found.append([False] * len(gt_subs))
            continue
        gt_found.append([gs in pred[label] for gs in gt_subs])

    # Compute precision
    pred_found = []
    for label, pred_subs in pred.items():
        if not pred_subs:
            continue
        if label not in gt:
            pred_found.append([False] * len(pred_subs))
            continue
        pred_found.append([ps in gt[label] for ps in pred_subs])

    recall_values = [sum(row) / len(row) for row in gt_found if row]
    precision_values = [sum(row) / len(row) for row in pred_found if row]

    recall = round(sum(recall_values) / len(recall_values), 3) if recall_values else 0.0
    precision = round(sum(precision_values) / len(precision_values), 3) if precision_values else 0.0

    if isinstance(precision, float) and math.isnan(precision):
        precision = 0.0

    scores["variable_recall"] = recall
    scores["variable_precision"] = precision

    # F1
    if precision + recall > 0:
        scores["variable_f1"] = round(2 * precision * recall / (precision + recall), 3)
    else:
        scores["variable_f1"] = 0.0

    return scores
