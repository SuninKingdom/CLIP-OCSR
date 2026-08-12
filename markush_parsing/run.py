#!/usr/bin/env python3
"""Markush Structure Image Parsing Pipeline.

Combines MinerU layout analysis, CLIP-OCSR recognition, and LLM-based
substituent extraction for complete Markush structure parsing.

Requires pre-computed MinerU layout outputs. Run MinerU first:
    mineru -p /path/to/images -o /path/to/mineru_outputs -b pipeline

Usage:
    # Run on a single image (prediction only)
    python run.py --input image.png --mineru-dir /path/to/mineru_outputs --llm deepseek

    # Run on a folder of images with labels for evaluation
    python run.py --input /path/to/images --labels labels.json --mineru-dir /path/to/mineru_outputs --llm deepseek

    # Evaluate saved results
    python run.py --evaluate results/deepseek/per_sample.jsonl

    # Generate concrete product SMILES from saved parsing results
    python run.py --instantiate-results results/deepseek/per_sample.jsonl

    # Test cropping on a single image
    python run.py --input image.png --mineru-dir /path/to/mineru_outputs --crop
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import glob
import platform
import sys

from rdkit import rdBase

from config import Config
from data_loader import SampleData, load_combined_label_entries
from image_crop import crop_markush_image
from evaluate import evaluate_single, compute_aggregate_metrics, save_results
from markush_instantiator import build_markush_instantiator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def configure_instantiation(config: Config, args, enabled: bool) -> Config:
    """Apply CLI concrete-product generation controls."""
    config.enable_instantiation = enabled
    config.use_fragment_library = not args.no_fragment_library
    if args.no_fragment_library:
        config.fragment_library_path = ""
    elif args.fragment_library:
        config.fragment_library_path = args.fragment_library
    config.instantiation_max_products = args.max_products
    config.instantiation_max_assignment_attempts = (
        args.max_assignment_attempts
    )
    config.instantiation_max_position_variants = args.max_position_variants
    config.instantiation_max_frequency_variants = args.max_frequency_variants
    config.instantiation_max_repeat_count = args.max_repeat_count
    config.instantiation_max_candidates_per_value = (
        args.max_candidates_per_value
    )
    return config


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_saved_value(row: dict, field_names: tuple[str, ...]):
    """Return the first populated saved-result field and its name."""
    for field_name in field_names:
        value = row.get(field_name)
        if value is not None and value != "":
            return value, field_name
    return None, "missing"


def collect_images(input_path: str) -> list[str]:
    """Collect image paths from a single file or a directory."""
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(input_path, ext)))
        paths.sort()
        return paths

    logger.error(f"Input path does not exist: {input_path}")
    return []


def build_samples(
    image_paths: list[str],
    labels_path: str | None,
    variable_labels_path: str | None = None,
) -> list[SampleData]:
    """Build samples and merge graphical/variable labels by image name."""
    # Load labels into a dict keyed by image_name
    labels_map = {}
    if labels_path and os.path.exists(labels_path):
        for entry in load_combined_label_entries(
            labels_path, variable_labels_path=variable_labels_path
        ):
            labels_map[entry["image_name"]] = entry

    samples = []
    for i, img_path in enumerate(image_paths):
        fname = os.path.basename(img_path)
        entry = labels_map.get(fname, {})
        samples.append(SampleData(
            id=i,
            image_name=fname,
            image_path=img_path,
            gt_smiles=entry.get("gt_smiles", ""),
            variables_gt=entry.get("variables", {}),
            pseudo_smiles_all=entry.get("pseudo_smiles_all", []),
        ))

    return samples


def cmd_run(args):
    """Run pipeline on input images."""
    # Import the model pipeline only for image processing. Offline
    # instantiation therefore does not require PyTorch or OCSR checkpoints.
    from pipeline import run_pipeline_batch

    config = Config(llm_provider=args.llm, mineru_output_dir=args.mineru_dir)
    configure_instantiation(config, args, enabled=args.instantiate)
    if args.output:
        config.output_dir = args.output

    image_paths = collect_images(args.input)
    if not image_paths:
        logger.error("No images found")
        return

    samples = build_samples(image_paths, args.labels, args.variable_labels)
    logger.info(f"Found {len(samples)} images, LLM={args.llm}")

    has_labels = any(s.gt_smiles for s in samples)
    if has_labels:
        logger.info("Labels loaded — will evaluate predictions")
    else:
        logger.info("No labels — prediction only")

    results = run_pipeline_batch(samples, config, resume=not args.no_resume, timing_path=args.timing)

    if has_labels:
        metrics = compute_aggregate_metrics(results)
        save_results(results, metrics, config.output_dir)
        print("\n" + "=" * 60)
        print("Aggregate Metrics:")
        print("=" * 60)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print("=" * 60)
    else:
        save_results(results, {}, config.output_dir)
        print(f"\nResults saved to {config.output_dir}")


def cmd_instantiate_results(args):
    """Instantiate saved predictions without rerunning any model or API."""
    input_path = os.path.abspath(args.instantiate_results)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Results file not found: {input_path}")

    if args.instantiation_output:
        output_path = os.path.abspath(args.instantiation_output)
    else:
        stem, _ = os.path.splitext(input_path)
        output_path = stem + "_instantiated.jsonl"
    if output_path == input_path:
        raise ValueError(
            "Instantiation output must differ from the source results file"
        )
    summary_path = os.path.splitext(output_path)[0] + "_summary.json"
    if summary_path == input_path:
        raise ValueError(
            "Instantiation summary path must differ from the source results file"
        )
    existing_outputs = [
        path for path in (output_path, summary_path) if os.path.exists(path)
    ]
    if existing_outputs and not args.overwrite_instantiation_output:
        raise FileExistsError(
            "Refusing to overwrite existing instantiation output(s): "
            + ", ".join(existing_outputs)
            + ". Pass --overwrite-instantiation-output to replace them."
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    config = Config(llm_provider=args.llm)
    configure_instantiation(config, args, enabled=True)
    instantiator = build_markush_instantiator(config)

    status_counts = Counter()
    failure_reasons = Counter()
    smiles_field_counts = Counter()
    variables_field_counts = Counter()
    rows_read = 0
    rows_written = 0
    malformed_rows = 0
    total_products = 0
    fully_enumerated = 0
    truncated = 0
    temporary_path = output_path + ".tmp"
    try:
        with open(input_path, "r", encoding="utf-8") as source, open(
            temporary_path, "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                rows_read += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed_rows += 1
                    logger.warning(
                        "Skipping malformed JSON at line %d: %s",
                        line_number,
                        exc,
                    )
                    continue

                try:
                    predicted_smiles, smiles_field = first_saved_value(
                        row,
                        (
                            "predicted_smiles",
                            "source_predicted_smiles",
                            "pred_smiles",
                        ),
                    )
                    predicted_variables, variables_field = first_saved_value(
                        row,
                        (
                            "predicted_variables",
                            "predicted_stable",
                            "source_predicted_variables",
                        ),
                    )
                    smiles_field_counts[smiles_field] += 1
                    variables_field_counts[variables_field] += 1
                    instantiation = instantiator.instantiate(
                        predicted_smiles,
                        predicted_variables,
                    )
                except Exception as exc:
                    logger.exception(
                        "Instantiation failed at line %d (%s)",
                        line_number,
                        row.get("image_name", "unknown sample"),
                    )
                    instantiation = {
                        "schema_version": instantiator.schema_version,
                        "status": "failed",
                        "failure_reason": "internal_instantiation_error",
                        "is_fully_enumerated": False,
                        "product_count": 0,
                        "products": [],
                        "errors": [f"{type(exc).__name__}: {exc}"],
                        "warnings": [],
                    }
                row["instantiation"] = instantiation
                destination.write(
                    json.dumps(row, ensure_ascii=False, default=str) + "\n"
                )
                rows_written += 1
                status = instantiation.get("status", "failed")
                status_counts[status] += 1
                failure_reason = instantiation.get("failure_reason")
                if failure_reason:
                    failure_reasons[failure_reason] += 1
                total_products += int(instantiation.get("product_count", 0))
                fully_enumerated += int(
                    bool(instantiation.get("is_fully_enumerated"))
                )
                truncated += int(bool(instantiation.get("truncated")))
        os.replace(temporary_path, output_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise

    summary = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_saved_prediction_instantiation",
        "command_argv": list(sys.argv),
        "source_results": input_path,
        "source_sha256": file_sha256(input_path),
        "output_results": output_path,
        "output_sha256": file_sha256(output_path),
        "rows_read": rows_read,
        "rows_written": rows_written,
        "malformed_rows_skipped": malformed_rows,
        "status_counts": dict(status_counts),
        "failure_reasons": dict(failure_reasons),
        "source_field_usage": {
            "pseudo_smiles": dict(smiles_field_counts),
            "variables": dict(variables_field_counts),
        },
        "fully_enumerated_samples": fully_enumerated,
        "truncated_samples": truncated,
        "total_unique_products": total_products,
        "settings": {
            "fragment_library": dict(instantiator.resolver.library_metadata),
            "limits": {
                "max_products": instantiator.limits.max_products,
                "max_assignment_attempts": (
                    instantiator.limits.max_assignment_attempts
                ),
                "max_position_variants": (
                    instantiator.limits.max_position_variants
                ),
                "max_frequency_variants": (
                    instantiator.limits.max_frequency_variants
                ),
                "max_repeat_count": instantiator.limits.max_repeat_count,
                "max_candidates_per_value": (
                    instantiator.resolver.max_candidates_per_value
                ),
            },
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "instantiation_schema": instantiator.schema_version,
        },
        "ground_truth_used": False,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Instantiated results: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Status counts: {dict(status_counts)}")
    print(f"Unique products written: {total_products}")


def cmd_evaluate(args):
    """Evaluate saved results."""
    path = args.evaluate
    if not os.path.exists(path):
        logger.error(f"Results file not found: {path}")
        return

    output_dir = os.path.dirname(path)

    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Always recompute saved scores so legacy result files cannot
                # silently retain the previous denominator or F1 definition.
                if entry.get("error"):
                    entry["scores"] = None
                elif entry.get("gt_smiles"):
                    entry["scores"] = evaluate_single(entry)
                results.append(entry)
            except json.JSONDecodeError:
                continue

    metrics = compute_aggregate_metrics(results)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    save_results(results, metrics, output_dir)


def cmd_crop(args):
    """Test MinerU layout analysis and cropping on a single image."""
    config = Config(llm_provider=args.llm, mineru_output_dir=args.mineru_dir)
    if args.output:
        config.output_dir = args.output

    image_paths = collect_images(args.input)
    if not image_paths:
        logger.error("No images found")
        return

    img_path = image_paths[0]
    save_dir = os.path.join(config.output_dir, "crop_test")
    crop = crop_markush_image(img_path, config, save_dir=save_dir)

    logger.info(f"Image: {img_path}")
    logger.info(f"Y threshold: {crop.y_threshold:.3f}")
    logger.info(f"Structure bbox: {crop.layout.structure_bbox}")
    logger.info(f"Text items: {len(crop.layout.text_items)}")
    logger.info(f"Saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Markush Structure Parsing Pipeline")

    parser.add_argument("--input", "-i", type=str, help="Input image or folder of images")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--labels", "-l", type=str, help="Labels JSON/CSV file (for evaluation)")
    parser.add_argument(
        "--variable-labels",
        type=str,
        help="Optional supplemental JSON file containing variable labels",
    )
    parser.add_argument("--mineru-dir", type=str, help="MinerU output directory (pre-computed layout)")
    parser.add_argument("--llm", choices=["deepseek", "mimo"], default="deepseek", help="LLM provider")
    parser.add_argument("--timing", type=str, help="Timing output file path (e.g. timing.txt)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")

    parser.add_argument("--evaluate", "-e", type=str, help="Evaluate saved results file")
    parser.add_argument("--crop", action="store_true", help="Test cropping only")
    parser.add_argument(
        "--instantiate",
        action="store_true",
        help="Also generate concrete product SMILES during image processing",
    )
    parser.add_argument(
        "--instantiate-results",
        type=str,
        help="Generate products from an existing per_sample.jsonl without API calls",
    )
    parser.add_argument(
        "--instantiation-output",
        type=str,
        help="Output JSONL for --instantiate-results (source is never overwritten)",
    )
    parser.add_argument(
        "--overwrite-instantiation-output",
        action="store_true",
        help="Explicitly replace an existing offline instantiation output",
    )
    parser.add_argument(
        "--fragment-library",
        type=str,
        help=(
            "Alternate description/name-to-fragment JSON file "
            "(defaults to the resource bundled with this project)"
        ),
    )
    parser.add_argument(
        "--no-fragment-library",
        action="store_true",
        help="Use only reviewed built-in mappings and deterministic rules",
    )
    parser.add_argument("--max-products", type=int, default=256)
    parser.add_argument("--max-assignment-attempts", type=int, default=10000)
    parser.add_argument("--max-position-variants", type=int, default=64)
    parser.add_argument("--max-frequency-variants", type=int, default=64)
    parser.add_argument("--max-repeat-count", type=int, default=100)
    parser.add_argument("--max-candidates-per-value", type=int, default=64)

    args = parser.parse_args()

    if args.fragment_library and args.no_fragment_library:
        parser.error(
            "--fragment-library and --no-fragment-library are mutually exclusive"
        )
    positive_limits = {
        "--max-products": args.max_products,
        "--max-assignment-attempts": args.max_assignment_attempts,
        "--max-position-variants": args.max_position_variants,
        "--max-frequency-variants": args.max_frequency_variants,
        "--max-candidates-per-value": args.max_candidates_per_value,
    }
    for option, value in positive_limits.items():
        if value < 1:
            parser.error(f"{option} must be at least 1")
    if args.max_repeat_count < 0:
        parser.error("--max-repeat-count cannot be negative")

    if args.instantiate_results:
        cmd_instantiate_results(args)
    elif args.evaluate:
        cmd_evaluate(args)
    elif args.crop:
        if not args.mineru_dir:
            parser.error("--mineru-dir is required with --crop")
        cmd_crop(args)
    elif args.input:
        if not args.mineru_dir:
            parser.error("--mineru-dir is required when running the pipeline")
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
