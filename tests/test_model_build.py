"""Tests for CLIP-OCSR model building."""

import unittest
import torch


class TestTransformerComponents(unittest.TestCase):
    """Test Transformer decoder components."""

    def test_layer_normalization(self):
        from clip_ocsr.models.transformer import LayerNormalization
        ln = LayerNormalization(512)
        x = torch.randn(2, 10, 512)
        out = ln(x)
        self.assertEqual(out.shape, (2, 10, 512))

    def test_feed_forward_block(self):
        from clip_ocsr.models.transformer import FeedForwardBlock
        ffn = FeedForwardBlock(512, 2048, 0.1)
        x = torch.randn(2, 10, 512)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 10, 512))

    def test_input_embeddings(self):
        from clip_ocsr.models.transformer import InputEmbeddings
        embed = InputEmbeddings(512, 83)
        x = torch.tensor([[1, 2, 3]])
        out = embed(x)
        self.assertEqual(out.shape, (1, 3, 512))

    def test_positional_encoding(self):
        from clip_ocsr.models.transformer import PositionalEncoding
        pe = PositionalEncoding(512, 256, 0.1)
        x = torch.randn(2, 10, 512)
        out = pe(x)
        self.assertEqual(out.shape, (2, 10, 512))

    def test_multi_head_attention(self):
        from clip_ocsr.models.transformer import MultiHeadAttentionBlock
        mha = MultiHeadAttentionBlock(512, 8, 0.1)
        x = torch.randn(2, 10, 512)
        out = mha(x, x, x, None)
        self.assertEqual(out.shape, (2, 10, 512))

    def test_decoder_block(self):
        from clip_ocsr.models.transformer import (
            MultiHeadAttentionBlock, FeedForwardBlock, DecoderBlock
        )
        self_attn = MultiHeadAttentionBlock(512, 8, 0.1)
        cross_attn = MultiHeadAttentionBlock(512, 8, 0.1)
        ffn = FeedForwardBlock(512, 2048, 0.1)
        block = DecoderBlock(512, self_attn, cross_attn, ffn, 0.1)

        x = torch.randn(2, 10, 512)
        enc_out = torch.randn(2, 16, 512)
        out = block(x, enc_out, None, None)
        self.assertEqual(out.shape, (2, 10, 512))


class TestCausalMask(unittest.TestCase):
    """Test causal mask generation."""

    def test_causal_mask_shape(self):
        from clip_ocsr.stage2.dataset import causal_mask
        mask = causal_mask(10)
        self.assertEqual(mask.shape, (1, 10, 10))

    def test_causal_mask_lower_triangular(self):
        from clip_ocsr.stage2.dataset import causal_mask
        mask = causal_mask(5)
        # First row should be all True (can attend to first position)
        self.assertTrue(mask[0, 0, 0])
        # Upper triangle should be False
        self.assertFalse(mask[0, 0, 1])
        self.assertFalse(mask[0, 1, 2])


if __name__ == '__main__':
    unittest.main()
