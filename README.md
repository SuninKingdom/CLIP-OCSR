# CLIP-OCSR: Bridging the Markush Gap in Optical Chemical Structure Recognition

[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8.18-blue)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/Dependency-RDKit-green)](https://www.rdkit.org/)

Official implementation of the paper: **"Bridging the Markush gap in optical chemical structure recognition via a CLIP-derived visual backbone and synthetic data generation"**.

CLIP-OCSR is a specialized encoder-decoder model for Optical Chemical Structure Recognition (OCSR). It focuses on the high-fidelity translation of complex chemical images into SMILES strings, with a particular emphasis on **Markush structures** ubiquitous in pharmaceutical patents.

> **Note:** This repository currently provides the **Evaluation Suite** used to assess model accuracy on SMILES and Pseudo-SMILES (Markush) predictions. Model weights and training datasets are not included in the current release.

---

## ✨ Model Highlights

* **CLIP-Derived Visual Backbone**: Utilizes a CLIP-RN50 encoder pretrained on chemical image-caption pairs for robust feature extraction.
* **Markush Structure Mastery**: Specifically optimized to handle complex structural variations (substituent, frequency, and position variations).
* **Deterministic Post-processing**: Employs an enumeration strategy to derive specific isomer sets from symbolic Pseudo-SMILES predictions.
* **MarkushGen Powered**: Developed using the MarkushGen toolkit to overcome the scarcity of annotated Markush images.

---

## 📊 Evaluation Suite

This toolkit provides the core logic for benchmarking OCSR models, especially those capable of generating Markush notations.

### Key Capabilities:
* **Pseudo-SMILES Validation**: Logic to verify if a predicted Markush string is chemically consistent with the ground truth.
* **Canonicalization**: Leveraging RDKit for robust molecular identity comparison.
* **Accuracy Metrics**: Scripts to calculate Top-1 accuracy and other performance indicators across diverse datasets.

---

## 🚀 Getting Started

### 1. Installation

Ensure you have [RDKit](https://www.rdkit.org/docs/Install.html) installed. We recommend using a Conda environment:

```bash
# Clone the repository
git clone [https://github.com/Sunin/CLIP-OCSR.git](https://github.com/Sunin/CLIP-OCSR.git)
cd CLIP-OCSR

# Install dependencies
pip install rdkit pandas numpy`CLIP-OCSR.onnx`

# Usage

## Inference

See `inference.py`

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
