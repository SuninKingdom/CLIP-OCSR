"""Generate sample training data for CLIP-OCSR.

Extracts ~300 SMILES from the existing training CSV and renders them as molecular
structure images at two resolutions:
- 224x224 for Stage 1 (CLIP pretraining)
- 512x512 for Stage 2 (OCSR fine-tuning)

Usage:
    python scripts/generate_sample_data.py
"""

import os
import csv
import sys
from pathlib import Path

import io
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image


def get_atom_counts(mol):
    """Count atoms by element type in a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        Dictionary mapping element symbols to counts.
    """
    counts = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def generate_caption(mol):
    """Generate a natural language caption for a molecule.

    Follows the exact format used in the original training data.

    Args:
        mol: RDKit molecule object.

    Returns:
        Caption string.
    """
    atom_counts = get_atom_counts(mol)
    total_atoms = sum(atom_counts.values())
    num_rings = mol.GetRingInfo().NumRings()

    # Build the "including X atoms" part
    element_parts = []
    for symbol in ['O', 'N', 'S', 'F', 'Cl', 'Br', 'I', 'P']:
        if symbol in atom_counts:
            count = atom_counts[symbol]
            atom_word = "atom" if count == 1 else "atoms"
            element_parts.append(f"{count} {symbol} {atom_word}")

    elements_str = ", ".join(element_parts)
    ring_word = "ring" if num_rings == 1 else "rings"

    if elements_str:
        caption = f"Non-Markush molecular image. Contains {total_atoms} atoms, including {elements_str}. Contains {num_rings} {ring_word}."
    else:
        caption = f"Non-Markush molecular image. Contains {total_atoms} atoms. Contains {num_rings} {ring_word}."

    return caption


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    source_csv = "/data/qxhuang/ocsr/demo20/training_demo20_RandSel_sfp_multimol_filtered_maxlen255_aug5.csv"

    stage1_dir = project_root / "sample_data" / "stage1"
    stage2_dir = project_root / "sample_data" / "stage2"
    stage1_images = stage1_dir / "images"
    stage2_images = stage2_dir / "images"

    # Create directories
    stage1_images.mkdir(parents=True, exist_ok=True)
    stage2_images.mkdir(parents=True, exist_ok=True)

    # Read source CSV
    print(f"Reading source CSV from: {source_csv}")
    df = pd.read_csv(source_csv)

    # Take first 300 unique SMILES
    num_samples = 300
    smiles_list = df['SMILES'].dropna().unique()[:num_samples]
    print(f"Selected {len(smiles_list)} unique SMILES for sample data")

    # Generate images and CSV files
    stage1_rows = []  # file_path, caption_nl
    stage2_rows = []  # file_path, SMILES

    success_count = 0
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES {i}: {smiles[:50]}...")
            continue

        # Generate filenames
        filename = f"mol_{i:04d}.png"
        rel_path = f"images/{filename}"

        # Render Stage 1 image (224x224) - black and white
        try:
            drawer_224 = rdMolDraw2D.MolDraw2DCairo(224, 224)
            opts_224 = drawer_224.drawOptions()
            opts_224.useBWAtomPalette()
            drawer_224.DrawMolecule(mol)
            drawer_224.FinishDrawing()
            img_224 = Image.open(io.BytesIO(drawer_224.GetDrawingText()))
            img_224.save(stage1_images / filename)
        except Exception as e:
            print(f"Warning: Could not render molecule {i} at 224x224: {e}")
            continue

        # Render Stage 2 image (512x512) - black and white
        try:
            drawer_512 = rdMolDraw2D.MolDraw2DCairo(512, 512)
            opts_512 = drawer_512.drawOptions()
            opts_512.useBWAtomPalette()
            drawer_512.DrawMolecule(mol)
            drawer_512.FinishDrawing()
            img_512 = Image.open(io.BytesIO(drawer_512.GetDrawingText()))
            img_512.save(stage2_images / filename)
        except Exception as e:
            print(f"Warning: Could not render molecule {i} at 512x512: {e}")
            continue

        # Generate caption for Stage 1
        caption = generate_caption(mol)

        stage1_rows.append((rel_path, caption))
        stage2_rows.append((rel_path, smiles))
        success_count += 1

    # Write Stage 1 CSV
    stage1_csv = stage1_dir / "train.csv"
    with open(stage1_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'caption_nl'])
        for row in stage1_rows:
            writer.writerow(row)
    print(f"Stage 1 CSV written: {stage1_csv} ({len(stage1_rows)} rows)")

    # Write Stage 2 CSV
    stage2_csv = stage2_dir / "train.csv"
    with open(stage2_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'SMILES'])
        for row in stage2_rows:
            writer.writerow(row)
    print(f"Stage 2 CSV written: {stage2_csv} ({len(stage2_rows)} rows)")

    print(f"\nDone! Generated {success_count} sample molecules.")
    print(f"  Stage 1: {stage1_dir}")
    print(f"  Stage 2: {stage2_dir}")


if __name__ == "__main__":
    main()
