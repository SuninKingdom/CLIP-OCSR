"""Evaluation metrics for OCSR: InChI accuracy and Tanimoto similarity."""

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


def calculate_acc(mol_pred, mol_ref):
    """Calculate exact match accuracy using InChI comparison.

    Args:
        mol_pred: Predicted RDKit molecule object.
        mol_ref: Reference RDKit molecule object.

    Returns:
        1 if InChI strings match, 0 otherwise.
    """
    ide = 0
    if mol_pred is None:
        return ide
    try:
        inchi1 = Chem.inchi.MolToInchi(mol_pred)
        inchi2 = Chem.inchi.MolToInchi(mol_ref)
        if inchi1 is not None and inchi2 is not None:
            if inchi1 == inchi2:
                ide = 1
    except Exception as e:
        print(f"Error processing molecule: {e}. Skipping this entry.")
        return ide
    return ide


def calculate_tanimoto_similarity(mol_pred, mol_ref):
    """Calculate Tanimoto similarity using Morgan fingerprints.

    Args:
        mol_pred: Predicted RDKit molecule object.
        mol_ref: Reference RDKit molecule object.

    Returns:
        Tanimoto similarity score (0.0 to 1.0), or 0 if either molecule is None.
    """
    if mol_pred is not None and mol_ref is not None:
        fp1 = AllChem.GetMorganFingerprint(mol_pred, 2)
        fp2 = AllChem.GetMorganFingerprint(mol_ref, 2)
        tanimoto_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return tanimoto_similarity
    else:
        return 0
