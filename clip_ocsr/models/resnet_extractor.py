"""Extract ResNet50 encoder from a fine-tuned CLIP checkpoint.

This module reconstructs the ResNet50 visual encoder from a CLIP checkpoint
without requiring the 'clip' package at runtime. It uses the local
clip_model.py (a copy of OpenAI's CLIP architecture) to rebuild the model.
"""

import torch
from clip_ocsr.models.clip_model import build_model


class ResNetEncoder(torch.nn.Module):
    """
    ResNet50 encoder extracted from CLIP's ModifiedResNet.
    Takes image tensors and outputs feature maps.
    """

    def __init__(self, clip_visual: torch.nn.Module):
        super().__init__()
        self.stem = torch.nn.Sequential(
            clip_visual.conv1,
            clip_visual.bn1,
            clip_visual.relu1,
            clip_visual.conv2,
            clip_visual.bn2,
            clip_visual.relu2,
            clip_visual.conv3,
            clip_visual.bn3,
            clip_visual.relu3,
            clip_visual.avgpool
        )
        self.layer1 = clip_visual.layer1
        self.layer2 = clip_visual.layer2
        self.layer3 = clip_visual.layer3
        self.layer4 = clip_visual.layer4

    def forward(self, x: torch.Tensor):
        x = x.type(self.stem[0].weight.dtype)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def extract_resnet_from_finetuned_checkpoint(ckpt_path: str, device: torch.device = None):
    """
    Extract the ResNet50 encoder from a fine-tuned CLIP checkpoint.

    Args:
        ckpt_path: Path to the fine-tuned CLIP checkpoint file.
        device: Target device. Defaults to CUDA if available, else CPU.

    Returns:
        ResNetEncoder with fine-tuned weights.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    print(f"Loading finetuned checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model_state_dict = checkpoint.get('model_state_dict')

    if not model_state_dict:
        raise ValueError("Checkpoint file does not contain 'model_state_dict' key.")

    # Build the full CLIP model and inject weights
    clip_model = build_model(model_state_dict).to(device).eval()
    print("Successfully built CLIP model and applied finetuned weights.")

    # Extract the ResNet50 visual encoder
    resnet50_encoder = ResNetEncoder(clip_model.visual).to(device).eval()

    # Ensure float32
    if resnet50_encoder.stem[0].weight.dtype != torch.float32:
        resnet50_encoder.float()

    return resnet50_encoder
