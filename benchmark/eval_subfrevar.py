import re
import copy
from collections import Counter
from rdkit import Chem
from rdkit.Chem import MolStandardize


exclude_items = ['C@H', 'C@', 'C@@H', 'C@@', 'N+', 'O-', 'C-']


def extract_and_filter_bracket_content(s, exclude_items):
    contents = re.findall(r'\[(.*?)\]', s)
    
    filtered_contents = [content for content in contents if content not in exclude_items]
    
    return filtered_contents


def compare_subfrevar(s1, s2, exclude_items):
    """
    s1: Predicted SMILES
    s2: True SMILES
    """
    if not (isinstance(s1, str) and isinstance(s2, str)):
        print("Error: Input format is not correct")
        return 0
    if s1 == s2:
        return 1
    else:
        vars1 = extract_and_filter_bracket_content(s1, exclude_items)
        vars2 = extract_and_filter_bracket_content(s2, exclude_items)
        if Counter(vars1) != Counter(vars2):
            return 0
        else:
            print(f"vars1: {vars1}")
            vs = list(set(vars1))
            print(f"vars: {vs}")
            # Replace [xx] with 'P(C * n)' instead of 'CC(C)C' * n, etc. in SMILES  
            for i in range(len(vs)):
                s1 = s1.replace("[" + vs[i] + "]", "P" + "(" + "C" * (i + 1) + ")")
                s2 = s2.replace("[" + vs[i] + "]", "P" + "(" + "C" * (i + 1) + ")")
            print(f"s1: {s1}")
            print(f"s2: {s2}")
            
            mol1 = Chem.MolFromSmiles(s1)
            mol2 = Chem.MolFromSmiles(s2)
            if mol1 is None or mol2 is None:
                return 0
            
            standardizer = MolStandardize.Standardizer()
            mol1 = standardizer.standardize(mol1)
            mol2 = standardizer.standardize(mol2)

            canonical_smiles1 = Chem.MolToSmiles(mol1)
            canonical_smiles2 = Chem.MolToSmiles(mol2)

            print(f"canonical_smiles1: {canonical_smiles1}")
            print(f"canonical_smiles2: {canonical_smiles2}")
            
            if canonical_smiles1 == canonical_smiles2:
                return 1
            else:
                return 0


"""
s1 = "[Rp][N+](C)(C)[C@@H](CCOC1=C2C=CC=CC2=CC=C1)C1=CC=CC=C1"

s2 = "[Rp][N+]([C@@H](CCO/C1=C/C=C\C2=CC=CC=C12)C1=CC=CC=C1)(C)C"

exclude_items = ['C@H', 'C@', 'C@@H', 'C@@', 'N+']

print(compare_subfrevar(s1, s2, exclude_items))
"""
