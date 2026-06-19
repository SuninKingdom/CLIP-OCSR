#!/usr/bin/env python3
"""Markush Structure Image Parsing Pipeline.

Combines MinerU layout analysis, CLIP-OCSR recognition, and LLM-based
substituent extraction for complete Markush structure parsing.

Usage:
    # Run on a single image (prediction only)
    python run.py --input image.png --output results/deepseek --llm deepseek

    # Run on a folder of images with labels for evaluation
    python run.py --input /path/to/images --labels labels.json --output results/deepseek --llm deepseek

    # Evaluate saved results
    python run.py --evaluate results/deepseek/per_sample.jsonl

    # Test cropping on a single image
    python run.py --input image.png --crop
"""

import argparse
import json
import logging
import os
import glob

from config import Config
from data_loader import SampleData, load_dataset
from image_crop import crop_markush_image
from pipeline import run_pipeline_single, run_pipeline_batch
from evaluate import evaluate_single, compute_aggregate_metrics, save_results
from llm_client import SubstituentExtractor
from ocsr_client import OCSRClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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


def build_samples(image_paths: str, labels_path: str | None) -> list[SampleData]:
    """Build SampleData list. If labels provided, match by image_name."""
    # Load labels into a dict keyed by image_name
    labels_map = {}
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            for entry in json.load(f):
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
        ))

    return samples


def cmd_run(args):
    """Run pipeline on input images."""
    config = Config(llm_provider=args.llm)
    if args.output:
        config.output_dir = args.output

    image_paths = collect_images(args.input)
    if not image_paths:
        logger.error("No images found")
        return

    samples = build_samples(image_paths, args.labels)
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
                if "scores" not in entry and not entry.get("error"):
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
    config = Config(llm_provider=args.llm)
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
    parser.add_argument("--labels", "-l", type=str, help="Labels JSON file (for evaluation)")
    parser.add_argument("--llm", choices=["deepseek", "mimo"], default="deepseek", help="LLM provider")
    parser.add_argument("--timing", type=str, help="Timing output file path (e.g. timing.txt)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")

    parser.add_argument("--evaluate", "-e", type=str, help="Evaluate saved results file")
    parser.add_argument("--crop", action="store_true", help="Test cropping only")

    args = parser.parse_args()

    if args.evaluate:
        cmd_evaluate(args)
    elif args.crop:
        cmd_crop(args)
    elif args.input:
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
