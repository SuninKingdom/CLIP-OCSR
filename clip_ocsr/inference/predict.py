"""Inference module for CLIP-OCSR: Predict SMILES from molecular structure images.

Usage:
    # Using only Stage 2 checkpoint (recommended):
    python -m clip_ocsr.inference.predict --image path/to/image.png \
        --weights path/to/stage2_model.pt

    # Using both Stage 1 and Stage 2 checkpoints:
    python -m clip_ocsr.inference.predict --image path/to/image.png \
        --weights path/to/stage2_model.pt --clip_ckpt path/to/stage1_clip.pt
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from tokenizers import Tokenizer

from clip_ocsr.stage2.model import SmilesModel, build_smilesmodel
from clip_ocsr.stage2.encoder import OcsrEncoder
from clip_ocsr.models.resnet_extractor import ResNetEncoder
from clip_ocsr.models.transformer import (
    InputEmbeddings, PositionalEncoding, MultiHeadAttentionBlock,
    FeedForwardBlock, DecoderBlock, Decoder, ProjectionLayer
)
from clip_ocsr.stage2.dataset import causal_mask
from clip_ocsr.utils.abbrev_group import abbrevgroup2smiles


def load_model(weights_path, clip_ckpt_path=None, tokenizer_path="assets/tokenizer_smiles.json",
               tgt_seq_len=256, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):
    """Load a trained SmilesModel from checkpoint.

    Args:
        weights_path: Path to the Stage 2 model checkpoint.
        clip_ckpt_path: Optional path to the Stage 1 CLIP checkpoint.
            If not provided, the model will be built with random initialization
            and then the Stage 2 weights will be loaded (which include the
            fine-tuned visual encoder).
        tokenizer_path: Path to the SMILES tokenizer JSON file.
        tgt_seq_len: Target sequence length.
        d_model: Model dimension.
        N: Number of decoder layers.
        h: Number of attention heads.
        dropout: Dropout rate.
        d_ff: Feed-forward dimension.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer_tgt = Tokenizer.from_file(tokenizer_path)

    if clip_ckpt_path:
        # Build model with Stage 1 CLIP checkpoint
        model = build_smilesmodel(
            clip_ckpt_path, False, tokenizer_tgt.get_vocab_size(),
            tgt_seq_len, d_model, N, h, dropout, d_ff
        )
    else:
        # Build model without Stage 1 checkpoint (will be overwritten by Stage 2 weights)
        # Use a temporary random initialization
        from clip_ocsr.models.resnet_extractor import ResNetEncoder
        from torchvision.models import resnet50
        resnet = resnet50(pretrained=False)
        resnet_clip = ResNetEncoder(resnet)
        img_encoder = OcsrEncoder(resnet_clip, False)

        tgt_embed = InputEmbeddings(d_model, tokenizer_tgt.get_vocab_size())
        tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
        decoder_blocks = []
        for _ in range(N):
            decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            decoder_block = DecoderBlock(d_model, decoder_self_attention_block,
                                         decoder_cross_attention_block, feed_forward_block, dropout)
            decoder_blocks.append(decoder_block)
        decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
        projection_layer = ProjectionLayer(d_model, tokenizer_tgt.get_vocab_size())
        model = SmilesModel(img_encoder, decoder, tgt_embed, tgt_pos, projection_layer)

    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    return model, tokenizer_tgt


def preprocess_image(image_path):
    """Load and preprocess a molecular structure image for inference.

    The image is:
    1. Converted to RGB
    2. Center-cropped/padded to 512x512 with white background
    3. Converted to a tensor

    Args:
        image_path: Path to the input image.

    Returns:
        Image tensor of shape (1, 3, 512, 512).
    """
    image = Image.open(image_path).convert("RGB")

    # Convert to grayscale for bounding box detection
    gray = image.convert("L")
    gray_np = np.array(gray)

    # Find bounding box of content (invert to find dark content on white background)
    inverted = 255 - gray_np
    rows = np.any(inverted > 10, axis=1)
    cols = np.any(inverted > 10, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Crop with some padding
        pad = 10
        rmin = max(0, rmin - pad)
        rmax = min(image.height, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(image.width, cmax + pad)
        image = image.crop((cmin, rmin, cmax, rmax))

    # Center-pad to 512x512
    target_size = 512
    ratio = min(target_size / image.width, target_size / image.height)
    new_w = int(image.width * ratio)
    new_h = int(image.height * ratio)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Create white background and paste centered
    background = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    background.paste(image, (offset_x, offset_y))

    # Convert to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(background).unsqueeze(0)


def greedy_decode_single(model, image_tensor, tokenizer, max_len, device):
    """Run greedy decoding on a single image.

    Args:
        model: SmilesModel (not DDP-wrapped).
        image_tensor: Input image tensor (1, 3, 512, 512).
        tokenizer: SMILES tokenizer.
        max_len: Maximum decoding length.
        device: Target device.

    Returns:
        Decoded token IDs.
    """
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device)

    sos_idx = tokenizer.token_to_id('<sos>')
    eos_idx = tokenizer.token_to_id('<eos>')

    decoder_input = torch.full((1, 1), sos_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        encoder_output = model.encode(image_tensor)

        for _ in range(max_len):
            decoder_mask = causal_mask(decoder_input.size(1)).to(device)
            decoder_output = model.decode(encoder_output, None, decoder_input, decoder_mask)
            prob = model.project(decoder_output[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.long()

            if next_word.item() == eos_idx:
                break

            decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)

    return decoder_input.squeeze(0)


def predict_smiles(image_path, model, tokenizer, device, abbrev_group_path=None):
    """End-to-end SMILES prediction from an image.

    Args:
        image_path: Path to the input molecular structure image.
        model: Trained SmilesModel.
        tokenizer: SMILES tokenizer.
        device: Target device.
        abbrev_group_path: Optional path to abbreviation group JSON for expansion.

    Returns:
        Predicted SMILES string.
    """
    image_tensor = preprocess_image(image_path)
    token_ids = greedy_decode_single(model, image_tensor, tokenizer, 256, device)
    smiles = tokenizer.decode(token_ids.detach().cpu().numpy()).replace(" ", "")

    if abbrev_group_path:
        smiles = abbrevgroup2smiles(smiles, abbrev_group_path)

    return smiles


def main():
    parser = argparse.ArgumentParser(description="CLIP-OCSR Inference: Predict SMILES from molecular images")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, required=True, help="Path to Stage 2 model checkpoint (.pt)")
    parser.add_argument("--clip_ckpt", type=str, default=None, help="Optional: Path to Stage 1 CLIP checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="assets/tokenizer_smiles.json",
                        help="Path to SMILES tokenizer JSON")
    parser.add_argument("--abbrev_group", type=str, default="assets/abbrev_group.json",
                        help="Path to abbreviation group JSON")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    print("Loading model...")
    model, tokenizer = load_model(args.weights, args.clip_ckpt, args.tokenizer)
    model.to(device)

    print(f"Predicting SMILES for: {args.image}")
    smiles = predict_smiles(args.image, model, tokenizer, device, args.abbrev_group)
    print(f"Predicted SMILES: {smiles}")


if __name__ == "__main__":
    main()
