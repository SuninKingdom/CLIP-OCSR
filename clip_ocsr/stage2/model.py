"""CLIP-OCSR model: SmilesModel combining image encoder with Transformer decoder."""

import torch
import torch.nn as nn

from clip_ocsr.models.resnet_extractor import extract_resnet_from_finetuned_checkpoint
from clip_ocsr.stage2.encoder import OcsrEncoder
from clip_ocsr.models.transformer import (
    InputEmbeddings, PositionalEncoding, MultiHeadAttentionBlock,
    FeedForwardBlock, DecoderBlock, Decoder, ProjectionLayer
)


class SmilesModel(nn.Module):
    """Image-to-SMILES translation model.

    Combines a CLIP-RN50 image encoder with a Transformer decoder
    to generate SMILES strings from molecular structure images.

    Args:
        img_encoder: Image encoder (OcsrEncoder).
        decoder: Transformer decoder.
        tgt_embed: Target token embeddings.
        tgt_pos: Target positional encoding.
        projection_layer: Linear projection to vocabulary size.
    """

    def __init__(self, img_encoder, decoder: Decoder, tgt_embed: InputEmbeddings,
                 tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.img_encoder = img_encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src_img):
        return self.img_encoder(src_img)

    def decode(self, encoder_output: torch.Tensor, src_mask: None,
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

    def forward(self, src_img, tgt, tgt_mask, return_last_token=False):
        encoder_output = self.encode(src_img)
        decoder_output = self.decode(encoder_output, None, tgt, tgt_mask)
        return_last_token = bool(return_last_token)
        if return_last_token:
            projected_output = self.project(decoder_output[:, -1])
        else:
            projected_output = self.project(decoder_output)
        return projected_output


def initialize_stage2_model(smiles_model: SmilesModel):
    """Initialize Stage 2 model parameters.

    - Conv1x1 weights: Kaiming normal init
    - Decoder embedding/projection: Normal init (mean=0, std=0.02)
    - Decoder attention/feed-forward: Xavier uniform init
    - Biases: Zeros
    """
    print("Starting Stage 2 model parameter initialization...")

    total_params = 0
    initialized_params = 0
    frozen_params = 0

    for name, param in smiles_model.named_parameters():
        total_params += param.numel()

        # 1. Initialize conv1x1 layer
        if 'img_encoder.conv1x1' in name:
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                initialized_params += param.numel()
            else:
                nn.init.zeros_(param)
                initialized_params += param.numel()

        # 2. Skip all other backbone parameters
        elif 'img_encoder' in name:
            frozen_params += param.numel()
            continue

        # 3. Initialize Decoder related parameters
        elif any(keyword in name for keyword in ['decoder', 'tgt_embed', 'tgt_pos', 'projection_layer']):
            if param.dim() > 1:
                if 'embed' in name or 'projection' in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    nn.init.xavier_uniform_(param)
                initialized_params += param.numel()
            else:
                if 'bias' in name:
                    nn.init.zeros_(param)
                    initialized_params += param.numel()

    print(f"Initialization completed:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Initialized: {initialized_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,}")
    print(f"  - Initialization ratio: {initialized_params/total_params*100:.1f}%")


def build_smilesmodel(clip_ckpt_path, trainable, tgt_vocab_size: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048, output_channels: int = 512) -> SmilesModel:
    """Build the complete SmilesModel.

    Args:
        clip_ckpt_path: Path to the Stage 1 CLIP checkpoint.
        trainable: Whether the ResNet50 encoder should be trainable.
        tgt_vocab_size: Target vocabulary size.
        tgt_seq_len: Target sequence length.
        d_model: Model dimension.
        N: Number of decoder layers.
        h: Number of attention heads.
        dropout: Dropout rate.
        d_ff: Feed-forward dimension.
        output_channels: Output channels for conv1x1 layer.

    Returns:
        Initialized SmilesModel.
    """
    # Create CLIP image encoder
    resnet_clip = extract_resnet_from_finetuned_checkpoint(str(clip_ckpt_path))
    img_encoder = OcsrEncoder(resnet_clip, trainable)

    # Create embedding layer
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layer
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block,
                                     decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create decoder and projection layer
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create complete model
    smiles_model = SmilesModel(img_encoder, decoder, tgt_embed, tgt_pos, projection_layer)

    # Initialize parameters
    initialize_stage2_model(smiles_model)

    return smiles_model
