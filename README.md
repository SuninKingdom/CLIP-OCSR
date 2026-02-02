# CLIP-OCSR: Bridging the Markush Gap in Optical Chemical Structure Recognition

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](#) 
[![Zenodo](https://img.shields.io/badge/Weights-Zenodo-blue)](https://zenodo.org/your-link-here)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8.18-blue)](https://www.python.org/downloads/release/python-3818/)

Official implementation of the paper: **"Bridging the Markush gap in optical chemical structure recognition via a CLIP-derived visual backbone and synthetic data generation"**.

CLIP-OCSR is an advanced encoder-decoder framework for Optical Chemical Structure Recognition (OCSR). By leveraging a **CLIP-derived visual backbone** with domain-adaptive pretraining, it significantly enhances the recognition of general chemical structures and complex **Markush structures** with positional variations.


---

## ✨ Key Features

* **CLIP-derived Backbone**: Initialized with a visual encoder pretrained on chemical image-description pairs, providing superior feature extraction over traditional CNNs.
* **Markush-Specialized**: Specifically optimized to handle Markush structures, bridging a significant gap in current OCSR tools.
* **Post-processing Toolkit**: Integrated logic to resolve abbreviated groups and enumerate positional isomers from generated pseudo-SMILES.
* **ONNX Inference**: Optimized for high-performance inference on both CPU and NVIDIA GPUs.

---

## 🚀 Getting Started

### 1. Prerequisites
* **Python 3.8.18**
* **CUDA 11.8+** and **cuDNN 8.9+** (For GPU acceleration)

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/YourUsername/CLIP-OCSR.git](https://github.com/YourUsername/CLIP-OCSR.git)
cd CLIP-OCSR

# Create and activate the environment
conda create -n clip-ocsr python=3.8.18
conda activate clip-ocsr

# Install dependencies
pip install -r requirements.txt


# 3. Download Weights

Download the following assets from [Zenodo](https://zenodo.org/) and place them in the root directory:

- `demo18_aug4_resnet_ep45.onnx`: Pretrained model weights.
- `tokenizer_demo18_MinSim_smiles.json`: Specialized tokenizer for chemical structures.

# Usage

## Quick Inference

You can run inference on a single image using the provided script. The model outputs the recognized SMILES (or pseudo-SMILES for Markush structures).

```bash
python src/inference.py --input ./data/example.png
```

## Handling Positional Variations

For Markush structures with positional variations, CLIP-OCSR generates a template string. The post-processing logic in `src/process.py` uses `data/abbrev_group.json` to expand these into full chemical representations.

# Project Structure

```text
CLIP-OCSR/
├── src/
│   ├── inference.py        # Main inference script (CLI supported)
│   └── process.py          # Image preprocessing and greedy decoding logic
├── data/
│   ├── abbrev_group.json   # Group abbreviation and Markush mapping
│   └── example.png        # Sample chemical structure image
├── requirements.txt      # Python dependency list
├── LICENSE           # Apache 2.0 License
└── README.md           # Documentation
```

# Citation

If you find this work helpful in your research, please cite our paper:

```bibtex
@article{yourname2026clipocsr,
  title={Bridging the Markush gap in optical chemical structure recognition via ...},
  author={Your Name and Others},
  journal={Your Target Journal},
  year={2026}
}
```

# License

- Code: This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- Weights: The model weights available on Zenodo are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
