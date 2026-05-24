"""Expand abbreviated functional groups in SMILES strings."""

import json
import re


def abbrevgroup2smiles(smiles, abbrev_group_json_file):
    """Expand abbreviated functional groups in a SMILES string.

    Replaces bracketed group names (e.g., [COOH], [Me], [Ph]) with their
    full SMILES representations using a lookup table.

    Args:
        smiles: Input SMILES string potentially containing abbreviations.
        abbrev_group_json_file: Path to the JSON file mapping group names
            to their SMILES expansions.

    Returns:
        SMILES string with all abbreviations expanded.
    """
    with open(abbrev_group_json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    parts = smiles.split('.')

    processed_parts = []
    for part in parts:
        matches = re.findall(r'\[([^\[\]]+)\]', part)
        result = part
        for match in matches:
            if match in json_data:
                json_value = json_data[match]
                # Use index [1] for groups at the start of a fragment, [0] otherwise
                if result.startswith('[' + match + ']'):
                    replacement = json_value[1]
                else:
                    replacement = json_value[0]
                result = result.replace('[' + match + ']', str(replacement), 1)
        processed_parts.append(result)

    final_result = '.'.join(processed_parts)
    return final_result
