import json
import logging
import os
import time

from tqdm import tqdm

from config import Config
from data_loader import SampleData
from image_crop import crop_markush_image
from llm_client import SubstituentExtractor
from ocsr_client import OCSRClient
from evaluate import evaluate_single

logger = logging.getLogger(__name__)


def run_pipeline_single(
    sample: SampleData,
    ocsr: OCSRClient,
    extractor: SubstituentExtractor,
    config: Config,
) -> dict:
    """Run the full pipeline on a single sample.

    Flow:
    1. MinerU layout analysis -> crop structure/text, get extracted chemical image + OCR text
    2. OCSR: use MinerU extracted chemical image (if available) or cropped structure
    3. Substituent extraction: use MinerU OCR text -> text LLM (no vision needed)
    """
    result = {
        "image_name": sample.image_name,
        "gt_smiles": sample.gt_smiles,
        "gt_variables": sample.variables_gt,
        "predicted_smiles": None,
        "predicted_variables": None,
        "y_threshold": None,
        "time_seconds": None,
        "error": None,
    }

    t_start = time.time()

    try:
        # Step 1: Crop with MinerU layout analysis
        img_stem = os.path.splitext(sample.image_name)[0]
        save_dir = os.path.join(config.output_dir, img_stem)
        crop = crop_markush_image(sample.image_path, config, save_dir=save_dir)
        result["y_threshold"] = crop.y_threshold

        # Step 2: OCSR — prefer MinerU extracted chemical image, fallback to cropped structure
        ocsr_image = crop.mineru_structure_image or crop.structure_image
        result["predicted_smiles"] = ocsr.predict_smiles(ocsr_image)

        # Step 3: Substituent extraction — use OCR text (not image)
        result["predicted_variables"] = extractor.extract_substituents(crop.ocr_text)

        # Step 4: Evaluate (only if ground truth available)
        if sample.gt_smiles:
            result["scores"] = evaluate_single(result)

        # Save per-image result (no ground truth)
        per_image = {k: v for k, v in result.items() if not k.startswith("gt_")}
        result_path = os.path.join(save_dir, 'result.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(per_image, f, ensure_ascii=False, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error processing {sample.image_name}: {e}")
        result["error"] = str(e)

    result["time_seconds"] = round(time.time() - t_start, 2)
    return result


def load_checkpoint(output_dir: str) -> set:
    """Load already-processed image names from checkpoint file."""
    checkpoint_path = os.path.join(output_dir, "per_sample.jsonl")
    processed = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed.add(entry["image_name"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return processed


def append_result(result: dict, output_dir: str):
    """Append a single result to the checkpoint file."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "per_sample.jsonl")
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


def save_timing(results: list, timing_path: str):
    """Save timing information to a file."""
    os.makedirs(os.path.dirname(timing_path) or ".", exist_ok=True)
    with open(timing_path, "w", encoding="utf-8") as f:
        f.write("image_name\ttime_seconds\n")
        for r in results:
            t = r.get("time_seconds", 0)
            f.write(f"{r['image_name']}\t{t}\n")

        total = sum(r.get("time_seconds", 0) for r in results)
        f.write(f"\ntotal\t{round(total, 2)}\n")
        f.write(f"count\t{len(results)}\n")
        if results:
            f.write(f"avg\t{round(total / len(results), 2)}\n")


def run_pipeline_batch(
    samples: list, config: Config, resume: bool = True, timing_path: str | None = None
) -> list:
    """Run pipeline on all samples with checkpointing."""
    ocsr = OCSRClient(config)
    extractor = SubstituentExtractor(config)

    processed_names = load_checkpoint(config.output_dir) if resume else set()

    if processed_names:
        logger.info(f"Resuming: {len(processed_names)} samples already processed")

    results = []

    # Load existing results if resuming
    if processed_names:
        checkpoint_path = os.path.join(config.output_dir, "per_sample.jsonl")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    samples_to_process = [s for s in samples if s.image_name not in processed_names]

    if not samples_to_process:
        logger.info("All samples already processed")
        if timing_path:
            save_timing(results, timing_path)
        return results

    logger.info(f"Processing {len(samples_to_process)} samples...")

    for sample in tqdm(samples_to_process, desc="Processing"):
        result = run_pipeline_single(sample, ocsr, extractor, config)
        results.append(result)
        append_result(result, config.output_dir)

        # Rate limiting
        time.sleep(1)

    if timing_path:
        save_timing(results, timing_path)
        logger.info(f"Timing saved to {timing_path}")

    return results
