import re
from collections import Counter
import rdkit
from rdkit import Chem
from eval_subfrevar import *

def compare_posvar(s1, s2, s, exclude_items):
  """
    s1: Predicted SMILES
    s2: Labels
    s: SMILES set contains all possible substituent positions
  """
    if s1 == s2:
        return 1
    else:
        pattern = r'\[[^\[\]]*\$\]'

        rv1 = re.findall(pattern, s1)
        rv2 = re.findall(pattern, s2)

        if Counter(rv1) == Counter(rv2):
            s1 = s1.replace('$', '')
            print(f's1: {s1}')
            s_list = s.split(';')
            for s_e in s_list:
                print(f's_e: {s_e}')
                if compare_subfrevar(s1, s_e, exclude_items) == 1:
                    return 1
            return 0
        else:
            return 0
