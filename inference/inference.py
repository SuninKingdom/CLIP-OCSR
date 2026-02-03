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





