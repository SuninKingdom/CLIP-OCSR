# CLIP-OCSR

**Bridging the Markush Gap in Optical Chemical Structure Recognition via a CLIP-Derived Visual Backbone and Synthetic Data Generation**

A two-stage framework for recognizing chemical structures from images using CLIP-based visual encoders and Transformer decoders.

## Overview

CLIP-OCSR converts molecular structure images into SMILES strings through a two-stage training pipeline:

1. **Stage 1: Domain-Adaptive Pretraining** - Fine-tunes the CLIP-RN50 visual encoder on 1 million synthetic chemical structure image-caption pairs using contrastive learning, bridging the domain gap between natural images and chemical structures.

2. **Stage 2: Supervised Fine-tuning** - Combines the adapted CLIP-RN50 encoder with a 6-layer Transformer decoder and trains end-to-end on image-SMILES pairs for chemical structure recognition.

## Live Demo

We provide a ready-to-use web interface hosted on Hugging Face Spaces. You can upload chemical images (including those with complex Markush variations) and experience the model's recognition capabilities firsthand without any local setup.

👉 [Try CLIP-OCSR on Hugging Face Spaces](https://huggingface.co/spaces/Sunin/CLIP-OCSR)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- conda or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/SuninKingdom/CLIP-OCSR.git
cd clip_ocsr

# Create conda environment
conda create -n clip_ocsr python=3.10
conda activate clip_ocsr

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install OpenAI CLIP (required for Stage 1)
pip install git+https://github.com/openai/CLIP.git

# Install the package
pip install -e .
```

### CLIP Pretrained Weights

Stage 1 requires OpenAI's CLIP RN50 pretrained weights. Download them with:

```bash
# Option 1: Automatic download via CLIP package
python -c "import clip; clip.load('RN50', device='cpu')"

# Option 2: Direct download
wget https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
```

Then update `pretrained_path` in `configs/stage1_pretrain.yaml`.

## Quick Start

### 1. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates ~300 sample molecular images in `sample_data/` for testing the pipeline.

### 2. Stage 1: CLIP Domain-Adaptive Pretraining

```bash
# Single GPU
torchrun --nproc_per_node=1 -m clip_ocsr.stage1.train --config configs/stage1_pretrain.yaml

# Multi-GPU (e.g., 2 GPUs)
torchrun --nproc_per_node=2 -m clip_ocsr.stage1.train --config configs/stage1_pretrain.yaml
```

### 3. Stage 2: OCSR Fine-tuning

Update `clip_ckpt_path` in `configs/stage2_finetune.yaml` to point to your Stage 1 checkpoint, then:

```bash
# Single GPU
torchrun --nproc_per_node=1 -m clip_ocsr.stage2.train --config configs/stage2_finetune.yaml

# Multi-GPU (e.g., 2 GPUs)
torchrun --nproc_per_node=2 -m clip_ocsr.stage2.train --config configs/stage2_finetune.yaml
```

### 4. Inference

```bash
python -m clip_ocsr.inference.predict \
    --image path/to/molecule.png \
    --weights checkpoints/stage2/stage2_clip_ocsr_45.pt
```

## Project Structure

```
clip_ocsr/
├── clip_ocsr/                  # Main Python package
│   ├── stage1/                 # Stage 1: CLIP pretraining
│   │   ├── train.py            # Training script
│   │   ├── config.py           # Configuration
│   │   └── dataset.py          # Dataset class
│   ├── stage2/                 # Stage 2: OCSR fine-tuning
│   │   ├── train.py            # Training script
│   │   ├── config.py           # Configuration
│   │   ├── dataset.py          # Dataset class
│   │   ├── model.py            # SmilesModel architecture
│   │   └── encoder.py          # OcsrEncoder (ResNet + Conv1x1)
│   ├── models/                 # Shared model components
│   │   ├── clip_model.py       # CLIP architecture (OpenAI)
│   │   ├── transformer.py      # Transformer decoder
│   │   └── resnet_extractor.py # ResNet extraction from CLIP
│   ├── inference/              # Inference code
│   │   └── predict.py          # PyTorch inference pipeline
│   ├── evaluation/             # Evaluation metrics
│   │   └── metrics.py          # InChI accuracy + Tanimoto
│   └── utils/                  # Utilities
│       ├── seed.py             # Reproducibility
│       └── abbrev_group.py     # Abbreviated group expansion
├── configs/                    # YAML configuration files
├── assets/                     # Tokenizer, abbreviation data, and sample SMILES for Stage 1/2
├── scripts/                    # Helper scripts
├── sample_data/                # Sample training data
└── tests/                      # Unit tests
```

## Model Architecture

### Stage 1: CLIP Domain-Adaptive Pretraining

- **Backbone**: CLIP-RN50 (ResNet-50 with anti-aliased convolutions)
- **Loss**: Symmetric cross-entropy (InfoNCE) contrastive loss
- **Optimizer**: AdamW (lr=2e-6, weight_decay=1e-4)
- **Scheduler**: Cosine annealing (T_max=60 epochs)
- **Input**: 224x224 RGB images + text captions

### Stage 2: OCSR Fine-tuning

- **Encoder**: CLIP-RN50 from Stage 1 + Conv1x1 (2048->512)
- **Decoder**: 6-layer Transformer (d_model=512, 8 heads, d_ff=2048)
- **Loss**: Cross-entropy with label smoothing (0.1)
- **Optimizer**: Adam (lr=1.5e-4)
- **Scheduler**: Cosine annealing (T_max=num_epochs/4)
- **Input**: 512x512 RGB images
- **Output**: SMILES strings (max length 256, 83-token vocabulary)

## Configuration

Training parameters are specified via YAML config files in `configs/`:

- `configs/stage1_pretrain.yaml` - Stage 1 hyperparameters
- `configs/stage2_finetune.yaml` - Stage 2 hyperparameters

Key parameters to set before training:

| Parameter | Config File | Description |
|-----------|------------|-------------|
| `pretrained_path` | stage1_pretrain.yaml | Path to OpenAI CLIP RN50 weights |
| `clip_ckpt_path` | stage2_finetune.yaml | Path to Stage 1 checkpoint |

## Citation

If you use CLIP-OCSR in your research, please cite:

```bibtex
@article{huang2026bridging,
  title={Bridging the Markush Gap in Optical Chemical Structure Recognition via a CLIP-Derived Visual Backbone and Synthetic Data Generation},
  author={Huang, Qixing and Chang, Haohua and Mao, Liyun and Shen, Zihao and Tian, Wenhao and Zheng, Meijuan and Wang, Jun and Li, Honglin},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  doi={<DOI>}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The CLIP model architecture (`clip_ocsr/models/clip_model.py`) is from [OpenAI CLIP](https://github.com/openai/CLIP) and is also licensed under MIT.

## Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for the pre-trained visual encoder
- [RDKit](https://www.rdkit.org/) for molecular processing and image rendering
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/) for SMILES tokenization
