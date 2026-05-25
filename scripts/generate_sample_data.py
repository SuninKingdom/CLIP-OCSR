"""Generate sample training data for CLIP-OCSR.

Reads SMILES from assets/sample_molecules.csv and renders them as molecular
structure images at two resolutions:
- 224x224 for Stage 1 (CLIP pretraining)
- 512x512 for Stage 2 (OCSR fine-tuning)

Usage:
    python scripts/generate_sample_data.py
"""

import csv
import io
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image


def get_atom_counts(mol):
    """Count atoms by element type in a molecule."""
    counts = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def generate_caption(mol):
    """Generate a natural language caption for a molecule.

    Follows the format: "Non-Markush molecular image. Contains X atoms,
    including Y O atoms, Z N atoms. Contains N rings."
    """
    atom_counts = get_atom_counts(mol)
    total_atoms = sum(atom_counts.values())
    num_rings = mol.GetRingInfo().NumRings()

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
    project_root = Path(__file__).parent.parent
    stage1_csv = project_root / "assets" / "sample_molecules_stage1.csv"
    stage2_csv = project_root / "assets" / "sample_molecules_stage2.csv"

    stage1_dir = project_root / "sample_data" / "stage1"
    stage2_dir = project_root / "sample_data" / "stage2"
    stage1_images = stage1_dir / "images"
    stage2_images = stage2_dir / "images"

    stage1_images.mkdir(parents=True, exist_ok=True)
    stage2_images.mkdir(parents=True, exist_ok=True)

    # Read SMILES from separate CSVs for each stage
    print(f"Reading Stage 1 SMILES from: {stage1_csv}")
    with open(stage1_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        smiles_list_stage1 = [row['SMILES'] for row in reader if row['SMILES'].strip()]
    print(f"Loaded {len(smiles_list_stage1)} SMILES for Stage 1")

    print(f"Reading Stage 2 SMILES from: {stage2_csv}")
    with open(stage2_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        smiles_list_stage2 = [row['SMILES'] for row in reader if row['SMILES'].strip()]
    print(f"Loaded {len(smiles_list_stage2)} SMILES for Stage 2")

    # Process Stage 1 molecules
    stage1_rows = []
    stage1_success = 0
    for i, smiles in enumerate(smiles_list_stage1):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse Stage 1 SMILES {i}: {smiles[:50]}...")
            continue

        filename = f"mol_{i:04d}.png"
        rel_path = f"images/{filename}"

        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(224, 224)
            opts = drawer.drawOptions()
            opts.useBWAtomPalette()
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            img = Image.open(io.BytesIO(drawer.GetDrawingText()))
            img.save(stage1_images / filename)
        except Exception as e:
            print(f"Warning: Could not render Stage 1 molecule {i} at 224x224: {e}")
            continue

        caption = generate_caption(mol)
        stage1_rows.append((rel_path, caption))
        stage1_success += 1

    # Process Stage 2 molecules
    stage2_rows = []
    stage2_success = 0
    for i, smiles in enumerate(smiles_list_stage2):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse Stage 2 SMILES {i}: {smiles[:50]}...")
            continue

        filename = f"mol_{i:04d}.png"
        rel_path = f"images/{filename}"

        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(512, 512)
            opts = drawer.drawOptions()
            opts.useBWAtomPalette()
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            img = Image.open(io.BytesIO(drawer.GetDrawingText()))
            img.save(stage2_images / filename)
        except Exception as e:
            print(f"Warning: Could not render Stage 2 molecule {i} at 512x512: {e}")
            continue

        stage2_rows.append((rel_path, smiles))
        stage2_success += 1

    # Write Stage 1 CSV
    stage1_train_csv = stage1_dir / "train.csv"
    with open(stage1_train_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'caption_nl'])
        for row in stage1_rows:
            writer.writerow(row)
    print(f"Stage 1 CSV written: {stage1_train_csv} ({len(stage1_rows)} rows)")

    # Write Stage 2 CSV
    stage2_train_csv = stage2_dir / "train.csv"
    with open(stage2_train_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'SMILES'])
        for row in stage2_rows:
            writer.writerow(row)
    print(f"Stage 2 CSV written: {stage2_train_csv} ({len(stage2_rows)} rows)")

    print(f"\nDone!")
    print(f"  Stage 1: {stage1_success} molecules -> {stage1_dir}")
    print(f"  Stage 2: {stage2_success} molecules -> {stage2_dir}")


if __name__ == "__main__":
    main()
