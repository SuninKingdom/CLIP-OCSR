"""Dataset for Stage 1: CLIP domain-adaptive pretraining."""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class CLIPFinetuneDataset(Dataset):
    """Dataset for CLIP fine-tuning with image-caption pairs.

    Reads a CSV file with 'file_path' and 'caption_nl' columns.
    Images are processed with the CLIP preprocess transform.
    Captions are tokenized using CLIP's built-in tokenizer.

    Args:
        df: DataFrame with 'file_path' and 'caption_nl' columns.
        base_image_dir: Root directory for image files.
        image_transform: CLIP preprocess transform.
        tokenizer: clip.tokenize function.
    """

    def __init__(self, df, base_image_dir, image_transform, tokenizer):
        self.df = df
        self.base_image_dir = base_image_dir
        self.image_transform = image_transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load and transform image
        image_path = os.path.join(self.base_image_dir, row['file_path'])
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback to the first sample
            row0 = self.df.iloc[0]
            image_path0 = os.path.join(self.base_image_dir, row0['file_path'])
            image = Image.open(image_path0).convert("RGB")
            image_tensor = self.image_transform(image)
            caption = row0['caption_nl']

        # Tokenize caption
        caption = row['caption_nl']
        text_tensor = self.tokenizer(caption, truncate=True)[0]

        return {
            "image": image_tensor,
            "text": text_tensor
        }


def get_tokenizer():
    """Return CLIP's tokenizer function."""
    import clip
    return clip.tokenize
