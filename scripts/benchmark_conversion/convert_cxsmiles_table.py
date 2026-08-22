#!/usr/bin/env python3
"""Batch-convert CXSMILES fields in CSV, JSON, or JSONL tables.

The converter preserves input order and columns, then appends an auditable
pseudo-SMILES candidate, conversion status, issue codes, complete diagnostics,
and metadata.  It is intended for MarkushGrapher-2 prediction exports and for
preparing CXSMILES label tables for subsequent manual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cxsmiles_to_pseudo_smiles import ConversionStatus, convert_cxsmiles


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Expected a JSON object at {path}:{line_number}"
                    )
                records.append(dict(value))
        return records
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            for key in ("results", "records", "data"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list) or not all(
            isinstance(record, dict) for record in value
        ):
            raise ValueError(f"Expected a JSON list of objects in {path}")
        return [dict(record) for record in value]
    raise ValueError(f"Unsupported input format {suffix!r}; use CSV, JSON, or JSONL")


def _output_fieldnames(records: Iterable[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def _image_name(value: Any, suffix: str) -> str:
    name = str(value).strip()
    if not name or not suffix:
        return name
    if Path(name).suffix.lower() in IMAGE_SUFFIXES:
        return name
    return name + suffix


def _missing_conversion(value: Any) -> dict[str, Any]:
    return {
        "input_cxsmiles": value,
        "status": "missing_prediction",
        "pseudo_smiles": None,
        "issues": [
            {
                "code": "MISSING_CXSMILES",
                "message": "The selected input field is empty.",
                "severity": "error",
                "feature": None,
                "atom_indices": [],
            }
        ],
        "metadata": {},
    }


def convert_records(
    records: Iterable[Mapping[str, Any]],
    *,
    cxsmiles_column: str,
    pseudo_column: str,
    image_column: str | None = None,
    output_image_column: str | None = None,
    image_suffix: str = "",
    exact_only: bool = False,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Convert records without dropping failures or changing their order."""

    converted: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    for row_number, source_record in enumerate(records, start=1):
        record = dict(source_record)
        if cxsmiles_column not in record:
            raise ValueError(
                f"Column {cxsmiles_column!r} is missing from input row {row_number}"
            )
        value = record.get(cxsmiles_column)
        if isinstance(value, str) and value.strip():
            conversion = convert_cxsmiles(value.strip()).to_dict()
        elif value is None or (isinstance(value, str) and not value.strip()):
            conversion = _missing_conversion(value)
        else:
            raise ValueError(
                f"Column {cxsmiles_column!r} at row {row_number} must be text or null"
            )

        status = str(conversion["status"])
        statuses[status] += 1
        candidate = conversion.get("pseudo_smiles") or ""
        if exact_only and status != ConversionStatus.SUCCESS.value:
            candidate = ""
        record[pseudo_column] = candidate
        record["conversion_status"] = status
        record["eligible_for_automatic_evaluation"] = (
            status == ConversionStatus.SUCCESS.value
        )
        issues = conversion.get("issues", [])
        record["conversion_issue_codes"] = ";".join(
            str(issue.get("code", "")) for issue in issues if issue.get("code")
        )
        record["conversion_issues"] = issues
        record["conversion_metadata"] = conversion.get("metadata", {})

        if image_column is not None:
            if image_column not in record:
                raise ValueError(
                    f"Column {image_column!r} is missing from input row {row_number}"
                )
            target_column = output_image_column or image_column
            record[target_column] = _image_name(record[image_column], image_suffix)
        converted.append(record)
    return converted, statuses


def _serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _write_csv(path: Path, records: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _output_fieldnames(records)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(path.parent),
            prefix=f".{path.name}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for record in records:
                writer.writerow({key: _serialize_cell(record.get(key)) for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-convert CXSMILES columns to auditable pseudo-SMILES."
    )
    parser.add_argument("input", type=Path, help="Input CSV, JSON, or JSONL table")
    parser.add_argument("output", type=Path, help="Output CSV table")
    parser.add_argument("--cxsmiles-column", default="cxsmiles")
    parser.add_argument("--pseudo-column", default="pseudo_smiles")
    parser.add_argument(
        "--image-column",
        help="Optional source identifier column to copy/normalize as an image name",
    )
    parser.add_argument(
        "--output-image-column",
        help="Output image column; defaults to --image-column when omitted",
    )
    parser.add_argument(
        "--image-suffix",
        default="",
        help="Suffix appended to identifiers that do not already name an image",
    )
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Leave review_required candidates blank (statuses are always retained)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must differ")
    records = _load_records(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")
    converted, statuses = convert_records(
        records,
        cxsmiles_column=args.cxsmiles_column,
        pseudo_column=args.pseudo_column,
        image_column=args.image_column,
        output_image_column=args.output_image_column,
        image_suffix=args.image_suffix,
        exact_only=args.exact_only,
    )
    _write_csv(output_path, converted, args.overwrite)
    print(
        f"Converted {len(converted)} rows to {output_path}; "
        f"statuses={dict(sorted(statuses.items()))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
