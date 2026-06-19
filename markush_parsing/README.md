# Multimodal Markush Structure Parsing

This module implements the multimodal Markush information extraction workflow described in the paper (Figure 9). It combines three components to parse complete Markush structures from document images:

1. **MinerU** — Layout analysis and OCR to separate the graphical scaffold from accompanying textual definitions
2. **CLIP-OCSR** — Backbone pseudo-SMILES generation from the cropped structure image
3. **LLM** — Structured variable definition extraction from OCR-derived text

The two outputs are combined to form a structured Markush representation that preserves both the graphical backbone information and the text-defined variable constraints.

## Prerequisites

- Python 3.10+
- CLIP-OCSR Stage 1 and Stage 2 checkpoints (see main [README](../README.md) for download instructions)
- A [MinerU](https://github.com/opendatalab/MinerU) account and API token
- An OpenAI-compatible LLM API key (e.g., DeepSeek, MiMo)

## Installation

From the repository root:

```bash
# Install CLIP-OCSR and its dependencies
pip install -e .

# Install additional dependencies for Markush parsing
pip install openai python-dotenv mineru-open-sdk

# Install PyTorch separately (see main README)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp markush_parsing/.env.example markush_parsing/.env
```

Edit `.env` with your settings:

```bash
# Model paths
STAGE1_CKPT_PATH=/path/to/stage1_clip_checkpoint.pt
STAGE2_CKPT_PATH=/path/to/stage2_ocsr_checkpoint.pt

# MinerU
MINERU_TOKEN=your_mineru_jwt_token_here

# LLM API (choose one)
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash
```

## Evaluation Data

The file `Complete_Markush_Representation.csv` contains source metadata for the 27 Markush descriptions (21 patent-derived, 6 journal-derived) used to evaluate the multimodal parsing workflow in the paper.

## Usage

### Run on a single image

```bash
python markush_parsing/run.py --input image.png --output results/ --llm deepseek
```

### Run on a folder of images with evaluation labels

```bash
python markush_parsing/run.py \
    --input /path/to/images \
    --labels labels.json \
    --output results/ \
    --llm deepseek
```

### Evaluate saved results

```bash
python markush_parsing/run.py --evaluate results/deepseek/per_sample.jsonl
```

### Test MinerU layout analysis only

```bash
python markush_parsing/run.py --input image.png --crop --output results/
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Markush Graphical Accuracy | Exact match accuracy |
| Variable Recall | Fraction of ground-truth substituents correctly predicted |
| Variable Precision | Fraction of predicted substituents that match ground truth |
| Variable F1 | Harmonic mean of recall and precision |

## Pipeline Architecture

```
Input Image
    |
    v
[MinerU] Layout analysis + OCR
    |                           |
    v                           v
Structure crop              OCR text
    |                           |
    v                           v
[CLIP-OCSR]                 [LLM]
Pseudo-SMILES               Variable definitions
    |                           |
    +---------------------------+
    |
    v
Structured Markush Representation
```

## Citation

If you use this module, please cite:

```bibtex
@article{clip_ocsr,
  title={Bridging the Markush Gap in Optical Chemical Structure Recognition via a CLIP-Derived Visual Backbone and Synthetic Data Generation},
  author={...},
  journal={...},
  year={2026}
}
```
