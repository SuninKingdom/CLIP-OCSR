"""Dataset for Stage 2: OCSR fine-tuning (image-to-SMILES)."""

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class SmilesDataset(Dataset):
    """Dataset for image-to-SMILES translation.

    Reads a CSV file with 'file_path' and 'SMILES' columns.
    Images are loaded and converted to tensors.
    SMILES strings are tokenized and padded to a fixed length.

    Args:
        dataframe: DataFrame with 'file_path' and 'SMILES' columns.
        img_folder: Root directory for image files.
        tokenizer_tgt: HuggingFace tokenizers Tokenizer object.
        seq_len: Maximum sequence length for decoder input.
        transform: Optional image transform. Defaults to ToTensor().
    """

    def __init__(self, dataframe, img_folder, tokenizer_tgt, seq_len, transform=None):
        super().__init__()
        self.dataframe = dataframe
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len
        self.img_folder = img_folder
        self.transform = transform

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("<sos>")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("<eos>")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("<pad>")], dtype=torch.int64)

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        img_path = self.img_folder + '/' + row[0]
        smiles = row[1]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        dec_input_tokens = self.tokenizer_tgt.encode(smiles).ids
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # Add SOS to the decoder input
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        # Add EOS to the label
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": image,
            "decoder_input": decoder_input,
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "tgt_text": smiles
        }


def causal_mask(size):
    """Generate a causal (autoregressive) mask for the decoder."""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
