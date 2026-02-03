# CLIP-OCSR: Bridging the Markush Gap in Optical Chemical Structure Recognition

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](#) 
[![Zenodo](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://zenodo.org/your-link-here)
[![Zenodo](https://img.shields.io/badge/Weights-Zenodo-blue)](https://zenodo.org/your-link-here)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8.18-blue)](https://www.python.org/downloads/release/python-3818/)

Official implementation of the paper: **"Bridging the Markush gap in optical chemical structure recognition via a CLIP-derived visual backbone and synthetic data generation"**.

CLIP-OCSR is a specialized encoder-decoder model for Optical Chemical Structure Recognition (OCSR), focusing on the high-fidelity translation of complex chemical images into SMILES strings. By combining a **domain-adaptively pretrained** CLIP vision encoder with a Transformer-based decoder, and leveraging our **MarkushGen** synthetic toolkit, it excels at recognizing **Markush structures** that challenge conventional methods.



---

## ✨ Key Features

* **CLIP-Derived Visual Backbone**: Utilizes a CLIP-RN50 encoder pretrained on chemical image-caption pairs for robust molecular feature extraction.
* **Markush Structure Mastery**: Specifically optimized to handle complex structural variations ubiquitous in pharmaceutical patents:
    * **Substituent & Frequency Variations**: Achieves significantly higher accuracy than existing SOTA methods.
    * **Position Variations**: Employing a deterministic post-processing enumeration strategy to exhaustively derive all isomer sets from symbolic predictions.
* **MarkushGen Powered**: Developed with the MarkushGen toolkit to synthesize diverse training data, overcoming the scarcity of annotated Markush images.
* **ONNX Inference**: High-performance inference scripts provided for both CPU and NVIDIA GPU (via CUDA).

---

## 🚀 Getting Started

### 1. Prerequisites

* **Python**: 3.8.18 (Recommended)
* **GPU Acceleration (Optional)**: To enable GPU inference, strict version alignment between **ONNX Runtime**, **CUDA**, and **cuDNN** is essential.
* **Verified Environment**: The following configuration has been successfully tested on our servers:
    * **ONNX Runtime**: `onnxruntime-gpu==1.19.2`
    * **CUDA**: 12.2
    * **cuDNN**: 9.6.0
* **Reference**: For other version combinations, please refer to the [Official Compatibility Matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

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
```
### 3. Download Weights

Download the model weight file from [Zenodo](https://zenodo.org/) and place it in the `weights` directory:

- `CLIP-OCSR.onnx`

# Usage

## Inference

See `inference.py`

```bash
import ctypes
import os
import onnxruntime as ort
from tokenizers import Tokenizer
import logging
from process import *
from rv_rearrange import *

# --- Environment Configuration ---
# Optional: Manually load cuDNN and CUDA runtime libraries if they are not in the system's default path.
# ctypes.CDLL('/your/path/cudnn-9.6.0/lib/libcudnn.so', mode=ctypes.RTLD_GLOBAL)
# ctypes.CDLL('/your/path/cuda-12.2/lib64/libcudart.so', mode=ctypes.RTLD_GLOBAL)

# Set the GPU device ID if multiple GPUs are available
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def predict(file_path, onnx_session, tokenizer_tgt, max_len):
    """
    Translates a chemical structure image into a machine-readable SMILES string.
    
    Args:
        file_path (str): Path to the input image.
        onnx_session (ort.InferenceSession): Initialized ONNX inference session.
        tokenizer_tgt (Tokenizer): Pre-trained SMILES tokenizer.
        max_len (int): Maximum sequence length for the decoder.
        
    Returns:
        tuple: (Image filename, predicted SMILES string)
    """
    # Extract the base filename for logging/output purposes
    img_name = os.path.basename(file_path)
    smiles = "error"
    
    try:
        # Step 1: Preprocess the image and convert it to a normalized tensor
        img_tensor = get_img_from_file(file_path).numpy()

        # Step 2: Run greedy decoding to generate the token sequence
        model_out = greedy_decode(onnx_session, img_tensor, tokenizer_tgt, max_len)

        # Step 3: Decode the token sequence into raw text (pseudo-SMILES)
        model_out_text = tokenizer_tgt.decode(model_out)
        smiles = model_out_text.replace(" ", "")

        # Step 4: Post-processing to replace abbreviated groups with full SMILES fragments
        # This is critical for Markush structures with positional variations.
        smiles = abbrevgroup2smiles(smiles, "abbrev_group.json")

        # Post-processing of CLIP-OCSR predictions for Markush structures with position variations
        if "$" in smiles:
            smiles = rearrange(smiles)

        return img_name, smiles

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except IOError as e:
        logging.error(f"Error reading the file: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during inference: {e}")

    # Return default values in case of failure
    return img_name, smiles

# --- Model Initialization ---

# Path to the pre-trained CLIP-OCSR ONNX model
onnx_model_path = "./weights/CLIP_OCSR.onnx"

# Path to the specialized SMILES tokenizer
tokenizer_path = "tokenizer_clip_ocsr.json"

# Execution Provider: Defaults to CUDA for GPU acceleration; fallbacks to CPU if CUDA is unavailable
# Note: Verified with CUDA 12.2 and cuDNN 9.6.0
provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Initialize the ONNX Runtime inference session
onnx_session = ort.InferenceSession(
    onnx_model_path,
    providers=provider
)

# Load the tokenizer from the JSON configuration
tokenizer_tgt = Tokenizer.from_file(tokenizer_path)

max_len = 256

# --- Example Usage ---
if __name__ == "__main__":
    # Path to the sample chemical structure image
    img_path = './examples/9213587.png' 
    
    # Run the prediction pipeline
    img_name, smiles = predict(img_path, onnx_session, tokenizer_tgt, max_len)
    
    # Print the final result
    print(f'Image: {img_name}')
    print(f'Predicted SMILES: {smiles}')

```

# Project Structure

```text
CLIP-OCSR/
├── src/
│   ├── inference.py        # Main inference script (supports CLI arguments)
│   └── process.py          # Preprocessing, greedy decoding, and Markush enumeration logic
├── data/
│   ├── abbrev_group.json   # Mapping for abbreviated groups and Markush positional variations
│   └── tokenizer.json      # Pre-trained pseudo-SMILES tokenizer
├── examples/               # Directory for sample images
│   └── example.png         # Sample chemical structure image
├── benchmark/              # Evaluation and comparative experiments
│   ├── eval.py             # Batch evaluation script for performance metrics
│   └── datasets/           # Standard test sets or download instructions
├── weights/                # Model weight storage
│   └── download_link.md    # Link to the pre-trained ONNX model on Zenodo
├── requirements.txt        # Python dependency list (locked to onnxruntime-gpu==1.19.2)
├── LICENSE                 # Apache 2.0 Open Source License
└── README.md               # Project documentation
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
