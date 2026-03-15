# CLIP-OCSR: Bridging the Markush Gap in Optical Chemical Structure Recognition

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Sunin/CLIP-OCSR)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8.18-blue)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/Dependency-RDKit-green)](https://www.rdkit.org/)

Official implementation of the paper: **"Bridging the Markush gap in optical chemical structure recognition via a CLIP-derived visual backbone and synthetic data generation"**.

CLIP-OCSR is a specialized encoder-decoder model for Optical Chemical Structure Recognition (OCSR). It focuses on the high-fidelity translation of complex chemical images into SMILES strings, with a particular emphasis on **Markush structures** ubiquitous in pharmaceutical patents.

---

## 🎮 Live Demo

We provide a **ready-to-use web interface** hosted on Hugging Face Spaces. You can upload chemical images (including those with complex Markush variations) and experience the model's recognition capabilities firsthand without any local setup.

👉 **[Try CLIP-OCSR on Hugging Face Spaces](https://huggingface.co/spaces/Sunin/CLIP-OCSR)**

---

## ✨ Model Highlights

* **CLIP-Derived Visual Backbone**: Utilizes a CLIP-RN50 encoder pretrained on chemical image-caption pairs for robust feature extraction.
* **Markush Structure Mastery**: Specifically optimized to handle complex structural variations (substituent, frequency, and position variations).
* **Deterministic Post-processing**: Employs an enumeration strategy to derive specific isomer sets from symbolic Pseudo-SMILES predictions.
* **MarkushGen Powered**: Developed using the MarkushGen toolkit to overcome the scarcity of annotated Markush images.

---

> **Note:** This repository currently provides the **Evaluation Suite** used to assess model accuracy on SMILES and Pseudo-SMILES (Markush) predictions. Model weights and training datasets are not included in the current release.

---

## 📊 Evaluation Suite

This toolkit provides the core logic for benchmarking OCSR models, especially those capable of generating Markush notations.

### Key Capabilities:
* **Pseudo-SMILES Validation**: Logic to verify if a predicted Markush string is chemically consistent with the ground truth.
* **Canonicalization**: Leveraging RDKit for robust molecular identity comparison.
* **Accuracy Metrics**: Scripts to compute exact match accuracy for both standard SMILES and Markush-specific Pseudo-SMILES

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/SuninKingdom/CLIP-OCSR.git
cd CLIP-OCSR

# Create and activate the environment
conda create -n clip-ocsr python=3.8.18
conda activate clip-ocsr

# Install dependencies
pip install rdkit==2022.09.1

```

### 2. Evaluation examples
```bash
cd benchmark

# 1. Standard OCSR Evaluation (non-Markush structures)
python eval.py

# 2. Markush Evaluation (Substituent & Frequency variations)
python eval_subfrevar.py

# 3. Markush Evaluation (Position variations)
python eval_psovar.py
```

# License

- Code: This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
