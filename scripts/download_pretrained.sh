#!/bin/bash
# Download OpenAI CLIP RN50 pretrained weights
#
# Usage:
#   bash scripts/download_pretrained.sh

set -e

echo "Downloading CLIP RN50 pretrained weights..."

# Option 1: Using Python (downloads to CLIP cache directory)
python -c "
import clip
print('Downloading CLIP RN50 model...')
model, preprocess = clip.load('RN50', device='cpu')
print('CLIP RN50 model downloaded successfully.')
print(f'Model cached at: ~/.cache/clip/')
"

echo ""
echo "Download complete!"
echo ""
echo "To use these weights, set 'pretrained_path' in configs/stage1_pretrain.yaml"
echo "to the cached path, typically:"
echo "  ~/.cache/clip/RN50.pt"
echo ""
echo "Or download directly:"
echo "  wget https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
