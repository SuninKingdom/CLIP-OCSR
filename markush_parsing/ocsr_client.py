"""OCSR client — CLIP-OCSR PyTorch model inference."""

import logging

import numpy as np
import torch
from PIL import Image, ImageOps, ImageChops
from torchvision import transforms
from tokenizers import Tokenizer

from clip_ocsr.stage2.model import build_smilesmodel
from clip_ocsr.stage2.dataset import causal_mask
from clip_ocsr.utils.abbrev_group import abbrevgroup2smiles

from config import Config

logger = logging.getLogger(__name__)

MAX_LEN = 256


def _preprocess_pil_image(image: Image.Image, target_size: int = 512) -> torch.Tensor:
    """Preprocess a PIL Image for CLIP-OCSR inference.

    Center-crop and pad to target_size, then convert to tensor.

    Args:
        image: PIL Image (RGB or will be converted to RGB).
        target_size: Target image size (default 512).

    Returns:
        Image tensor of shape (1, 3, target_size, target_size).
    """
    rgb_image = image.convert("RGB")
    grayscale = rgb_image.convert("L")
    inverted = ImageChops.invert(grayscale)
    bbox = inverted.getbbox()
    if bbox:
        cropped = rgb_image.crop(bbox)
    else:
        cropped = rgb_image
    padded = ImageOps.pad(cropped, (target_size, target_size), method=Image.BICUBIC, color=(255, 255, 255))
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(padded).unsqueeze(0)


# Module-level singleton (lazy init)
_predictor = None
_predictor_config_key = None


def _get_predictor(config: Config):
    global _predictor, _predictor_config_key
    config_key = (config.stage1_ckpt_path, config.stage2_ckpt_path, config.tokenizer_path)
    if _predictor is None or _predictor_config_key != config_key:
        tokenizer = Tokenizer.from_file(config.tokenizer_path)
        model = build_smilesmodel(
            config.stage1_ckpt_path, False, tokenizer.get_vocab_size(),
            tgt_seq_len=256, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048,
        )
        state = torch.load(config.stage2_ckpt_path, map_location='cpu')
        model.load_state_dict(state['model_state_dict'])
        model.eval()

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        model.to(device)

        _predictor = (model, tokenizer, device)
        _predictor_config_key = config_key
        logger.info(f"CLIP-OCSR model loaded on {device}")

    return _predictor


class OCSRClient:
    """CLIP-OCSR inference using PyTorch model."""

    def __init__(self, config: Config):
        self.config = config
        self.abbrev_group_path = config.abbrev_group_path

    def predict_smiles(self, image: Image.Image) -> str | None:
        """Convert chemical structure image to SMILES string.

        Returns None if inference fails.
        """
        try:
            model, tokenizer, device = _get_predictor(self.config)
            image_tensor = _preprocess_pil_image(image).to(device)

            sos_idx = tokenizer.token_to_id('<sos>')
            eos_idx = tokenizer.token_to_id('<eos>')

            decoder_input = torch.full((1, 1), sos_idx, dtype=torch.long, device=device)

            with torch.no_grad():
                encoder_output = model.encode(image_tensor)
                for _ in range(MAX_LEN):
                    decoder_mask = causal_mask(decoder_input.size(1)).to(device)
                    decoder_output = model.decode(encoder_output, None, decoder_input, decoder_mask)
                    prob = model.project(decoder_output[:, -1])
                    _, next_word = torch.max(prob, dim=1)
                    next_word = next_word.long()

                    if next_word.item() == eos_idx:
                        break

                    decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)

            smiles = tokenizer.decode(decoder_input.squeeze(0).detach().cpu().numpy()).replace(" ", "")
            if self.abbrev_group_path:
                smiles = abbrevgroup2smiles(smiles, self.abbrev_group_path)
            return smiles

        except Exception as e:
            logger.error(f"OCSR inference failed: {e}")
            return None
