#!/usr/bin/env python
# coding: utf-8

import io
import re
import logging
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, MolStandardize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_ring_info(smiles):
    """
    Identifies rings and counts substitutable hydrogen positions in each ring.
    
    Args:
        smiles (str): SMILES string with placeholders.
        
    Returns:
        tuple: (list of ring atom indices, list of available substituent counts per ring atom)
    """
    # Temporarily replace placeholders with Carbon to allow RDKit parsing
    placeholders = re.findall(r'\[\w+\]', smiles)
    modified_smiles = smiles
    for ph in set(placeholders):
        modified_smiles = modified_smiles.replace(ph, 'C')
    
    try:
        mol_modified = Chem.MolFromSmiles(modified_smiles, sanitize=False)
        # Manually handle aromaticity for complex Markush templates
        Chem.SanitizeMol(mol_modified, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
    except Exception as e:
        raise ValueError(f"Invalid SMILES string after placeholder substitution: {e}")
    
    # Add explicit hydrogens to find available substitution sites
    mol = Chem.AddHs(mol_modified)
    ring_info = AllChem.GetSymmSSSR(mol)
    
    ring_atoms_list = []
    substituents_list = []
    
    for ring in ring_info:
        ring_atoms = list(ring)
        ring_atoms_list.append(ring_atoms)
        
        substituents = []
        for atom_idx in ring_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            # Count neighboring hydrogen atoms
            num_h = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1)
            substituents.append(num_h)
        substituents_list.append(substituents)
    
    return ring_atoms_list, substituents_list


def perform_rearrangement(mol, atom_idx, new_position_idx):
    """
    Relocates a substituent from its current position to a new ring position.
    
    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.
        atom_idx (int): Current index of the substituent atom.
        new_position_idx (int): Target index on the ring.
        
    Returns:
        rdkit.Chem.Mol: Modified molecule object.
    """
    editable_mol = Chem.EditableMol(mol)
    
    # Remove existing bonds for the substituent
    atom = mol.GetAtomWithIdx(atom_idx)
    for bond in atom.GetBonds():
        editable_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    
    # Add bond at the new position
    editable_mol.AddBond(new_position_idx, atom_idx, order=Chem.rdchem.BondType.SINGLE)
    
    new_mol = editable_mol.GetMol()
    new_mol.UpdatePropertyCache()
    return new_mol


def mol_block_to_smiles(mol_obj, original_placeholders, mapped_placeholders):
    """
    Converts a molecule block back to SMILES while restoring original Markush placeholders.
    """
    mol_block = Chem.MolToMolBlock(mol_obj)
    asterisk_indices = [i for i, char in enumerate(mol_block) if char == '*']
    
    # Identify R-group patterns like [R1001]
    r_pattern = r'\[R\d+\]'
    found_r_groups = re.findall(r_pattern, mol_block)
    
    mol_chars = list(mol_block)
    for i, r_group in enumerate(found_r_groups):
        if i < len(asterisk_indices):
            # Map back the encoded [R1xxx] to original placeholders
            idx_in_map = mapped_placeholders.index(r_group)
            mol_chars[asterisk_indices[i]] = mapped_placeholders[idx_in_map].strip('[]')

    restored_mol_block = "".join(mol_chars)
    mol = Chem.MolFromMolBlock(restored_mol_block, sanitize=False)
    smiles = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)

    # Convert numeric placeholders back to Markush [Rxxxx] format
    def replace_match(match):
        return f'[R{int(match.group(1)) + 1000}]'

    smiles = re.sub(r'\[(\d+)\*\:\d+\]', replace_match, smiles)
    return smiles.replace('~', '')


def get_r_atom_idx(symbol, smiles):
    """
    Locates the atom index of a specific Markush symbol in the SMILES string.
    """
    mol = Chem.MolFromSmarts(smiles)
    mol_block = Chem.MolToMolBlock(mol)
    pattern = re.compile(fr'V\s+(\d+)\s+{re.escape(symbol)}')
    match = pattern.search(mol_block)
    return int(match.group(1)) - 1 if match else None


def deduplicate_and_recover(smiles_list, original_placeholders, mapped_placeholders):
    """
    Removes chemically redundant structures and recovers original placeholder names.
    """
    # Identify positional variation tags (ending with $)
    pv_tags = [mapped_placeholders[i] for i, ph in enumerate(original_placeholders) if ph.endswith('$]')]
    
    modified_list = smiles_list.copy()
    for i in range(len(modified_list)):
        for p, tag in enumerate(pv_tags):
            # Replace tags with dummy alkyl chains for robust canonicalization
            modified_list[i] = modified_list[i].replace(tag, 'CC(C)C' * (p + 1))

    standardizer = MolStandardize.Standardizer()
    unique_indices = []
    seen_smiles = set()
    
    for i, smiles in enumerate(modified_list):
        mol = Chem.MolFromSmarts(smiles)
        if mol:
            mol = standardizer.standardize(mol)
            can_smi = Chem.MolToSmiles(mol)
            if can_smi not in seen_smiles:
                seen_smiles.add(can_smi)
                unique_indices.append(i)
    
    final_results = []
    for i in unique_indices:
        s = smiles_list[i]
        for orig, mapped in zip(original_placeholders, mapped_placeholders):
            s = s.replace(mapped, orig).replace("$", "")
        final_results.append(s)
        
    return list(set(final_results))


def rearrange_markush(smiles):
    """
    Main entry point: Enumerates all possible isomers for Markush structures 
    with positional variation placeholders (marked with '$').
    
    Args:
        smiles (str): Template SMILES string.
        
    Returns:
        list: All valid generated SMILES strings.
    """
    # Extract and map all placeholders to internal [R1xxx] format
    all_placeholders = list(set(re.findall(r"\[.*?\]", smiles, re.I)))
    clean_placeholders = [ph for ph in all_placeholders if '#' not in ph]
    
    if not clean_placeholders:
        return [smiles]

    mapped_placeholders = []
    internal_smiles = smiles
    for i, ph in enumerate(clean_placeholders):
        internal_id = f"[R{1001 + i}]"
        internal_smiles = internal_smiles.replace(ph, internal_id)
        mapped_placeholders.append(internal_id)

    # Identify variables requiring positional rearrangement
    pv_indices = [i for i, ph in enumerate(clean_placeholders) if ph.endswith('$]')]
    pv_internal_ids = [mapped_placeholders[i] for i in pv_indices]

    output_smiles = [internal_smiles]
    processing_queue = [internal_smiles]

    for pv_id in pv_internal_ids:
        next_round = []
        for current_smi in processing_queue:
            idx = get_r_atom_idx(pv_id, current_smi)
            ring_atoms_list, substituents_list = get_ring_info(current_smi)
            mol_obj = Chem.MolFromSmarts(current_smi)
            
            if idx is None: continue
            atom = mol_obj.GetAtomWithIdx(idx)

            for neighbor in atom.GetNeighbors():
                for r_idx, ring_atoms in enumerate(ring_atoms_list):
                    if neighbor.GetIdx() in ring_atoms:
                        avail_substituents = substituents_list[r_idx]
                        for sub_pos, count in enumerate(avail_substituents):
                            if count > 0:
                                # Generate new structural isomer
                                new_mol = perform_rearrangement(mol_obj, atom.GetIdx(), ring_atoms[sub_pos])
                                if new_mol:
                                    new_smi = mol_block_to_smiles(new_mol, clean_placeholders, mapped_placeholders)
                                    next_round.append(new_smi)
                                    output_smiles.append(new_smi)
        
        if next_round:
            processing_queue += next_round

    return deduplicate_and_recover(output_smiles, clean_placeholders, mapped_placeholders)


if __name__ == "__main__":
    # Example usage for Markush positional variation
    test_smiles = 'O=C(N([R6])O[R4])C1=NC=C(C(C[Y])=C1O[R3])C([X])C2=CC([R2$])=C(C=C2)[R1$]'
    results = rearrange_markush(test_smiles)
    
    print(f"Generated {len(results)} unique isomers:")
    for i, res in enumerate(results):
        print(f"{i+1}: {res}")
