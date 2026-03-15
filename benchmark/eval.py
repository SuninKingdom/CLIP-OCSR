from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


def calculate_acc(mol_pred, mol_ref):
    # Generate Inchi
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

    if mol_pred is not None and mol_ref is not None:
        fp1 = AllChem.GetMorganFingerprint(mol_pred, 2)  # Morgan fingerprint with radius 2
        fp2 = AllChem.GetMorganFingerprint(mol_ref, 2)

        tanimoto_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return tanimoto_similarity
    else:
        return 0


if __name__ == "__main__":
    s1 = "O=C1N(C)C2=CC=C(C=C2C3=NC(C)=NN13)C"
    s2 = "O=C1N(C)C2=CC=C(C=C2C3=NC(C)=NN13)C"
    mol1 = Chem.MolFromSmiles(s1)
    mol2 = Chem.MolFromSmiles(s1)

    print("-" * 30)
    print(f"S1: {s1}")
    print(f"S2: {s2}")
    
    acc = calculate_acc(mol1, mol2)
    print(f"Conclusion: {'MATCH' if acc == 1.0 else 'MISMATCH'}")
    print("-" * 30)
