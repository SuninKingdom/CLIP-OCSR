from PIL import Image, ImageOps, ImageChops
from torchvision import transforms
import numpy as np
import json
import re

# --- Image Preprocessing ---

def preprocess_and_center_image(image, target_size=(512, 512)):
    """
    Standardizes the input image by cropping to the chemical structure's 
    bounding box and centering it within a padded canvas.
    """
    # Ensure image is in RGB mode
    rgb_image = image.convert("RGB")
    
    # Convert to grayscale to identify the structure's boundaries
    grayscale_image = rgb_image.convert("L")
    
    # Invert the image so the structure (foreground) becomes non-zero/white
    inverted_image = ImageChops.invert(grayscale_image)
    
    # Find the bounding box of the structural region
    bbox = inverted_image.getbbox()
    
    if bbox:
        # Crop the image tightly to the structure
        cropped_image = rgb_image.crop(bbox)
    else:
        # Fallback to original if no bounding box is detected
        cropped_image = rgb_image
    
    # Pad and resize the image to the target dimensions while maintaining aspect ratio
    # BICUBIC interpolation is used for high-quality resizing
    centered_padded_image = ImageOps.pad(
        cropped_image, 
        target_size, 
        method=Image.BICUBIC, 
        color=(255, 255, 255)
    )

    return centered_padded_image

def get_img_from_file(file):
    """
    Reads an image file, applies preprocessing, and returns a normalized tensor.
    """
    # Open image file
    image = Image.open(file)
    
    # Apply centering and padding
    image = preprocess_and_center_image(image)

    # Define the transformation pipeline: convert to tensor (and normalize to [0, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Apply transformation and add the batch dimension (1, C, H, W)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

# --- Mask Generation for Transformer Decoder ---

def causal_mask_onnx(size):
    """
    Creates a causal (look-ahead) mask to prevent the decoder from 
    attending to future tokens during inference.
    """
    # Generate an upper triangular matrix (excluding the diagonal)
    mask = np.triu(np.ones((size, size)), k=1)
    
    # Convert to boolean mask where True indicates allowed positions (lower triangular)
    mask = mask == 0 
    
    # Reshape to (1, size, size) for compatibility with ONNX input
    mask = mask[None, :, :]

    return mask

# --- Decoding Logic ---

def greedy_decode(onnx_session, source, tokenizer_tgt, max_len):
    """
    Performs autoregressive greedy decoding to generate the SMILES sequence.
    """
    # Retrieve special token indices
    sos_idx = tokenizer_tgt.token_to_id('<sos>')
    eos_idx = tokenizer_tgt.token_to_id('<eos>')

    # Initialize the decoder sequence with the Start-of-Sequence token
    decoder_input = np.array([[sos_idx]], dtype=np.int64)

    while True:
        # Stop if the maximum sequence length is reached
        if decoder_input.shape[1] == max_len:
            break

        # Generate the causal mask for the current sequence length
        seq_len = decoder_input.shape[1]
        decoder_mask = causal_mask_onnx(seq_len)

        # Execute ONNX inference
        inputs = {
            'input': source,        # Encoded image features
            'tgt': decoder_input,   # Current decoded sequence
            'tgt_mask': decoder_mask # Mask for the current sequence
        }
        onnx_output = onnx_session.run(None, inputs)

        # Select the token with the highest probability from the last time step
        prob = onnx_output[0]
        next_word = np.argmax(prob[:, -1, :], axis=1)

        # Append the predicted token to the sequence
        decoder_input = np.concatenate([decoder_input, next_word.reshape(1, 1)], axis=1)

        # Terminate if the End-of-Sequence token is predicted
        if next_word[0] == eos_idx:
            break

    return decoder_input.squeeze(0)

# --- Cheminformatics Post-processing ---

def abbrevgroup2smiles(smiles, abbrev_group_json_file):
    """
    Maps abbreviated group placeholders in the predicted string to their 
    corresponding SMILES fragments based on a provided JSON mapping.
    This is essential for resolving Markush structure components.
    """
    # 1. Load the abbreviation mapping from JSON
    with open(abbrev_group_json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 2. Split the string by '.' to handle multi-component structures
    parts = smiles.split('.')

    processed_parts = []
    for part in parts:
        # Find all placeholders within brackets, e.g., [R1], [COOH]
        matches = re.findall(r'\[([^\[\]]+)\]', part)
        result = part
        for match in matches:
            # Check if the placeholder exists in the mapping database
            if match in json_data:
                json_value = json_data[match]
                
                # Context-aware replacement:
                # Use the second fragment if the group is at the start (affects connectivity)
                if result.startswith('[' + match + ']'):
                    replacement = json_value[1]
                else:
                    replacement = json_value[0]
                
                # Replace only the first occurrence to avoid erroneous bulk replacements
                result = result.replace('[' + match + ']', str(replacement), 1)
        
        processed_parts.append(result)

    # 4. Rejoin components into the final SMILES string
    final_result = '.'.join(processed_parts)
    return final_result

