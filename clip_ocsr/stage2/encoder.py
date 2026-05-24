"""OCSR image encoder: ResNet50 + Conv1x1 adapter."""

import torch
import torch.nn as nn


class OcsrEncoder(nn.Module):
    """Image encoder that adapts CLIP's ResNet50 features for the Transformer decoder.

    Takes the 2048-channel output from ResNet50, reduces it to 512 channels
    via a 1x1 convolution, and reshapes to a sequence of 256 tokens.

    Args:
        resnet_clip: ResNet50 encoder extracted from CLIP.
        trainable: Whether ResNet50 parameters should be trainable.
    """

    def __init__(self, resnet_clip, trainable) -> None:
        super().__init__()
        self.resnet_clip = resnet_clip
        self.conv1x1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)

        for p in self.resnet_clip.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        x = self.resnet_clip(x)  # (B, 2048, 16, 16) for 512x512 input
        x = self.conv1x1(x)      # (B, 512, 16, 16)
        x = x.view(x.size(0), 512, -1).permute(0, 2, 1)  # (B, 256, 512)
        return x
