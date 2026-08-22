#!/usr/bin/env python3
"""Conservative CXSMILES-to-pseudo-SMILES conversion.

This standalone utility is included for benchmark-output normalization.  In
this work it was used primarily to convert MarkushGrapher-2 CXSMILES
predictions into the pseudo-SMILES dialect evaluated by CLIP-OCSR.  The same
conversion logic supplied the initial automatic candidates for the USPTO-M
and M2S label-conversion audits; those public label resources were then
manually reviewed, so their final reviewed columns remain authoritative.

The pseudo-SMILES dialect used by CLIP-OCSR extends SMILES with three
display-oriented pseudo atoms:

* ``[R1]`` (and analogous labels) for variable atoms/groups;
* ``[(CH2)n]`` for a frequency-variation unit; and
* ``[R1$]`` for a position-variable group.

CXSMILES is more expressive than this dialect.  In particular, a CXSMILES
``m:`` feature stores the complete set of possible attachment atoms, whereas
the ``$`` marker does not.  Polymer S-groups may also contain arbitrary
multi-atom graphs that have no unique pseudo-SMILES spelling.  Consequently,
this module deliberately distinguishes exact conversions from review-only
candidates and unsupported inputs.  It never silently discards an extension.

Public API
----------
``convert_cxsmiles()`` returns a :class:`ConversionResult` with a status,
diagnostics, and (when defensible) a pseudo-SMILES string.
``cxsmiles_to_pseudo_smiles()`` is the strict convenience function: it returns
only an exact result unless ``allow_review=True`` is explicitly requested.

The parser follows the ChemAxon CXSMILES feature schema documented at:
https://docs.chemaxon.com/latest/formats_chemaxon-extended-smiles-and-smarts-cxsmiles-and-cxsmarts.html

This implementation requires RDKit.  Explicit hydrogens are retained while
parsing because CXSMILES atom indexes include explicit ``[H]`` atoms.
"""

import argparse
import html
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from rdkit import Chem, rdBase


CONVERTER_VERSION = "1.0.0"


class ConversionStatus(str, Enum):
    """Outcome of a CXSMILES-to-pseudo-SMILES conversion."""

    SUCCESS = "success"
    REVIEW_REQUIRED = "review_required"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


@dataclass(frozen=True)
class ConversionIssue:
    """Machine-readable conversion diagnostic."""

    code: str
    message: str
    severity: str = "error"
    feature: Optional[str] = None
    atom_indices: Tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        value = asdict(self)
        value["atom_indices"] = list(self.atom_indices)
        return value


@dataclass(frozen=True)
class ConversionResult:
    """Result returned by :func:`convert_cxsmiles`."""

    input_cxsmiles: str
    status: ConversionStatus
    pseudo_smiles: Optional[str] = None
    issues: Tuple[ConversionIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_exact(self) -> bool:
        """Whether the conversion completed without a review requirement."""

        return self.status is ConversionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "input_cxsmiles": self.input_cxsmiles,
            "status": self.status.value,
            "pseudo_smiles": self.pseudo_smiles,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": dict(self.metadata),
        }


class CXSMILESToPseudoSMILESError(ValueError):
    """Raised when the strict conversion API cannot return an exact result."""

    def __init__(self, result: ConversionResult):
        self.result = result
        codes = ", ".join(issue.code for issue in result.issues) or "UNKNOWN"
        message = (
            "This CXSMILES cannot be converted to pseudo-SMILES exactly; "
            "please inspect it manually ({}).".format(codes)
        )
        super().__init__(message)


@dataclass(frozen=True)
class _FrequencyGroup:
    group_type: str
    atom_indices: Tuple[int, ...]
    subscript: str
    connectivity: str
    head_bonds: str
    tail_bonds: str
    additional_data: str
    raw: str


@dataclass(frozen=True)
class _PositionVariation:
    source_atom: int
    target_atoms: Tuple[int, ...]
    raw: str


@dataclass(frozen=True)
class _ParsedCXSMILES:
    source: str
    core: str
    extension: str
    mol: Chem.Mol
    atom_spans: Tuple[Tuple[int, int], ...]
    labels: Mapping[int, str]
    frequency_groups: Tuple[_FrequencyGroup, ...]
    position_variations: Tuple[_PositionVariation, ...]
    unknown_extension_text: str


_POSITION_PATTERN = re.compile(
    r"(?:^|,)\s*(m:(\d+):(\d+(?:\.\d+)*))(?=,|$)"
)
_ATOM_LABEL_PATTERN = re.compile(r"\$[^$]*\$")
_NEXT_EXTENSION_FEATURE = re.compile(
    r",(?=\s*(?:[A-Za-z][A-Za-z0-9_]*:|[&@^][^,:]*:|\$))"
)


def _issue(
    code: str,
    message: str,
    severity: str = "error",
    feature: Optional[str] = None,
    atom_indices: Iterable[int] = (),
) -> ConversionIssue:
    return ConversionIssue(
        code=code,
        message=message,
        severity=severity,
        feature=feature,
        atom_indices=tuple(atom_indices),
    )


def _invalid_result(source: Any, issue: ConversionIssue) -> ConversionResult:
    return ConversionResult(
        input_cxsmiles="" if source is None else str(source),
        status=ConversionStatus.INVALID,
        issues=(issue,),
    )


def _unsupported_result(
    source: str,
    issues: Sequence[ConversionIssue],
    metadata: Optional[Mapping[str, Any]] = None,
) -> ConversionResult:
    return ConversionResult(
        input_cxsmiles=source,
        status=ConversionStatus.UNSUPPORTED,
        issues=tuple(issues),
        metadata=metadata or {},
    )


def _split_cxsmiles(source: str) -> Tuple[str, str]:
    """Split a CXSMILES into its SMILES core and extension body."""

    if "|" not in source:
        return source.strip(), ""

    first = source.find("|")
    last = source.rfind("|")
    if first == last or source[last + 1 :].strip():
        raise ValueError("The CXSMILES extension must be enclosed by one |...| block.")

    core = source[:first].strip()
    extension = source[first + 1 : last]
    if not core:
        raise ValueError("The CXSMILES has an empty SMILES core.")
    return core, extension


def _smiles_atom_spans(smiles: str) -> Tuple[Tuple[int, int], ...]:
    """Return source spans for atoms in SMILES parse order.

    Atom indexes in SMILES and CXSMILES are assigned in lexical parse order.
    This small lexer lets exact conversions preserve the input SMILES spelling,
    including stereochemical slash bonds and explicit hydrogens.
    """

    spans: List[Tuple[int, int]] = []
    index = 0
    while index < len(smiles):
        char = smiles[index]
        if char == "[":
            end = smiles.find("]", index + 1)
            if end < 0:
                raise ValueError("Unclosed bracket atom in the SMILES core.")
            spans.append((index, end + 1))
            index = end + 1
        elif char == "*":
            spans.append((index, index + 1))
            index += 1
        elif smiles.startswith("Cl", index) or smiles.startswith("Br", index):
            spans.append((index, index + 2))
            index += 2
        elif char in "BCNOPSFIbcnops":
            spans.append((index, index + 1))
            index += 1
        else:
            index += 1
    return tuple(spans)


def _mark_span(mask: List[bool], start: int, end: int) -> None:
    for index in range(start, end):
        mask[index] = True


def _parse_extension(
    extension: str,
) -> Tuple[Tuple[_FrequencyGroup, ...], Tuple[_PositionVariation, ...], str]:
    """Parse the CX features supported by this converter.

    A coverage mask is used so any unrecognized extension is reported instead
    of being discarded.  Empty separators emitted by some CXSMILES writers are
    harmless and are ignored.
    """

    mask = [False] * len(extension)

    for match in _ATOM_LABEL_PATTERN.finditer(extension):
        _mark_span(mask, match.start(), match.end())

    frequency_groups: List[_FrequencyGroup] = []
    search_from = 0
    while True:
        start = extension.find("Sg:", search_from)
        if start < 0:
            break
        # Commas inside the atom/head/tail lists are followed by numeric bond
        # or atom indexes.  A comma followed by a named feature starts the next
        # CX extension and therefore terminates this S-group.
        next_feature = _NEXT_EXTENSION_FEATURE.search(extension, start)
        end = next_feature.start() if next_feature is not None else len(extension)
        raw = extension[start:end].strip().rstrip(",")
        parts = raw.split(":")
        # MarkushGrapher predictions sometimes omit optional trailing empty
        # S-group fields (``Sg:n:12:a:ht``), while dataset labels retain them
        # (``Sg:n:12:a:ht:::``).  Both spellings carry the same information.
        fields = parts[1:] if parts and parts[0] == "Sg" else []
        additional_data = ":".join(fields[6:]).strip(":")
        fields.extend([""] * max(0, 7 - len(fields)))
        atom_text = fields[1].strip() if len(fields) > 1 else ""
        atoms: Tuple[int, ...]
        if atom_text and re.fullmatch(r"\d+(?:,\d+)*", atom_text):
            atoms = tuple(int(value) for value in atom_text.split(","))
        else:
            atoms = ()
        frequency_groups.append(
            _FrequencyGroup(
                group_type=fields[0].strip() if fields else "",
                atom_indices=atoms,
                subscript=html.unescape(fields[2]).strip(),
                connectivity=html.unescape(fields[3]).strip(),
                head_bonds=html.unescape(fields[4]).strip(),
                tail_bonds=html.unescape(fields[5]).strip(),
                additional_data=html.unescape(additional_data).strip(),
                raw=raw,
            )
        )
        _mark_span(mask, start, end)
        search_from = end + 1

    position_variations: List[_PositionVariation] = []
    for match in _POSITION_PATTERN.finditer(extension):
        targets = tuple(int(value) for value in match.group(3).split("."))
        position_variations.append(
            _PositionVariation(
                source_atom=int(match.group(2)),
                target_atoms=targets,
                raw=match.group(1),
            )
        )
        _mark_span(mask, match.start(1), match.end(1))

    unknown = "".join(
        char
        for index, char in enumerate(extension)
        if not mask[index] and char not in ", \t\r\n"
    )
    return tuple(frequency_groups), tuple(position_variations), unknown


def _parser_params() -> Chem.SmilesParserParams:
    params = Chem.SmilesParserParams()
    params.allowCXSMILES = True
    params.parseName = False
    params.removeHs = False
    return params


def _normalize_atom_label(label: str) -> Tuple[Optional[str], Optional[ConversionIssue]]:
    """Translate standardized CX atom-label encodings into display labels."""

    normalized = html.unescape(label).strip()
    if normalized.startswith("_AP") or normalized == "<AP>":
        return None, _issue(
            "ATTACHMENT_POINT_UNSUPPORTED",
            "CXSMILES attachment-point labels have no defined CLIP-OCSR "
            "pseudo-SMILES equivalent.",
            feature=label,
        )
    if re.fullmatch(r"_R\d+", normalized):
        normalized = normalized[1:]
    elif normalized.endswith("_p") and len(normalized) > 2:
        # ChemAxon encodes pseudo atoms such as X as X_p in the alias block.
        normalized = normalized[:-2]
    elif normalized == "Q_e":
        normalized = "Q"
    elif normalized == "star_e":
        return None, _issue(
            "UNNAMED_STAR_LABEL",
            "The generic CXSMILES star_e atom has no unambiguous pseudo-SMILES label.",
            feature=label,
        )

    if not normalized:
        return None, _issue(
            "EMPTY_ATOM_LABEL",
            "An empty CXSMILES atom label cannot form a pseudo atom.",
            feature=label,
        )
    if any(character in normalized for character in "[]|$"):
        return None, _issue(
            "UNSAFE_ATOM_LABEL",
            "The atom label contains pseudo-SMILES delimiter characters.",
            feature=label,
        )
    return normalized, None


def _parse_cxsmiles(source: Any) -> Tuple[Optional[_ParsedCXSMILES], List[ConversionIssue]]:
    issues: List[ConversionIssue] = []
    if not isinstance(source, str):
        issues.append(
            _issue("INVALID_INPUT_TYPE", "CXSMILES input must be a string.")
        )
        return None, issues
    source = source.strip()
    if not source or source.lower() == "null":
        issues.append(_issue("EMPTY_INPUT", "CXSMILES input is empty."))
        return None, issues

    try:
        core, extension = _split_cxsmiles(source)
        atom_spans = _smiles_atom_spans(core)
    except ValueError as error:
        issues.append(_issue("MALFORMED_CXSMILES", str(error)))
        return None, issues

    params = _parser_params()
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(source, params)
    if mol is None:
        issues.append(
            _issue(
                "RDKIT_PARSE_FAILED",
                "RDKit could not parse the CXSMILES while retaining explicit hydrogens.",
            )
        )
        return None, issues
    if len(atom_spans) != mol.GetNumAtoms():
        issues.append(
            _issue(
                "ATOM_INDEX_MISMATCH",
                "The SMILES lexer found {} atoms but RDKit found {}; CX atom "
                "indexes cannot be mapped safely.".format(
                    len(atom_spans), mol.GetNumAtoms()
                ),
            )
        )
        return None, issues

    labels: Dict[int, str] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("atomLabel"):
            continue
        normalized, label_issue = _normalize_atom_label(atom.GetProp("atomLabel"))
        if label_issue is not None:
            issues.append(
                _issue(
                    label_issue.code,
                    label_issue.message,
                    feature=label_issue.feature,
                    atom_indices=(atom.GetIdx(),),
                )
            )
        elif normalized is not None:
            labels[atom.GetIdx()] = normalized

    frequency_groups, position_variations, unknown = _parse_extension(extension)
    parsed = _ParsedCXSMILES(
        source=source,
        core=core,
        extension=extension,
        mol=mol,
        atom_spans=atom_spans,
        labels=labels,
        frequency_groups=frequency_groups,
        position_variations=position_variations,
        unknown_extension_text=unknown,
    )
    return parsed, issues


def _feature_metadata(parsed: _ParsedCXSMILES) -> Dict[str, Any]:
    return {
        "atom_count": parsed.mol.GetNumAtoms(),
        "atom_labels": [
            {"atom_index": index, "label": label}
            for index, label in sorted(parsed.labels.items())
        ],
        "frequency_groups": [
            {
                "atom_indices": list(group.atom_indices),
                "subscript": group.subscript,
                "connectivity": group.connectivity,
                "head_bonds": group.head_bonds,
                "tail_bonds": group.tail_bonds,
                "additional_data": group.additional_data,
            }
            for group in parsed.frequency_groups
        ],
        "position_variations": [
            {
                "source_atom": variation.source_atom,
                "target_atoms": list(variation.target_atoms),
            }
            for variation in parsed.position_variations
        ],
    }


def _frequency_unit_label(
    mol: Chem.Mol,
    atom_index: int,
    labels: Mapping[int, str],
    external_bonds: int,
) -> Tuple[Optional[str], Optional[ConversionIssue]]:
    if atom_index in labels:
        return labels[atom_index], None

    atom = mol.GetAtomWithIdx(atom_index)
    if atom.GetSymbol() == "*":
        return None, _issue(
            "UNLABELED_FREQUENCY_ATOM",
            "An unlabeled wildcard frequency unit cannot be named in pseudo-SMILES.",
            atom_indices=(atom_index,),
        )
    if atom.GetIsAromatic():
        return None, _issue(
            "AROMATIC_FREQUENCY_ATOM_UNSUPPORTED",
            "A single aromatic-atom repeat has no agreed pseudo-SMILES spelling.",
            atom_indices=(atom_index,),
        )
    if atom.GetFormalCharge() or atom.GetIsotope():
        return None, _issue(
            "DECORATED_FREQUENCY_ATOM_UNSUPPORTED",
            "Charged or isotopic frequency atoms require manual notation review.",
            atom_indices=(atom_index,),
        )

    symbol = atom.GetSymbol()
    if symbol not in {"B", "C", "N", "O", "P", "S", "Si"}:
        return None, _issue(
            "FREQUENCY_ELEMENT_UNSUPPORTED",
            "Element {} is not covered by the deterministic frequency-unit "
            "notation.".format(symbol),
            atom_indices=(atom_index,),
        )
    # A head-to-tail SRU with fewer than two explicit crossing bonds has one or
    # two implicit continuation bonds.  Those continuation bonds replace
    # hydrogens in the displayed repeat formula: terminal ``C`` is therefore
    # ``CH2`` in ``[(CH2)n]``, not the ``CH3`` implied by the plain SMILES core.
    implicit_continuations = max(0, 2 - external_bonds)
    hydrogen_count = max(
        0,
        atom.GetTotalNumHs(includeNeighbors=False) - implicit_continuations,
    )
    hydrogen_text = "" if hydrogen_count == 0 else "H"
    if hydrogen_count > 1:
        hydrogen_text += str(hydrogen_count)
    return symbol + hydrogen_text, None


def _external_bond_count(mol: Chem.Mol, atom_indices: Set[int]) -> int:
    return sum(
        1
        for bond in mol.GetBonds()
        if (bond.GetBeginAtomIdx() in atom_indices)
        != (bond.GetEndAtomIdx() in atom_indices)
    )


def _replace_atom_spans(
    core: str,
    spans: Sequence[Tuple[int, int]],
    replacements: Mapping[int, str],
) -> str:
    result = core
    for atom_index in sorted(replacements, reverse=True):
        start, end = spans[atom_index]
        result = result[:start] + replacements[atom_index] + result[end:]
    return result


def _position_carriers(
    parsed: _ParsedCXSMILES,
) -> Tuple[
    Optional[List[Tuple[_PositionVariation, str, Set[int]]]],
    List[ConversionIssue],
]:
    """Recognize the benchmark's explicit two-atom position carrier pattern.

    MarkushGrapher-2 commonly encodes a graphical ``R--(ring)`` position bond
    as a disconnected ``*C`` fragment: the wildcard carries the R label and
    either atom may be the ``m:`` source.  Other carrier graphs are not guessed.
    """

    mol = parsed.mol
    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    atom_to_fragment = {
        atom_index: set(fragment)
        for fragment in fragments
        for atom_index in fragment
    }
    carriers: List[Tuple[_PositionVariation, str, Set[int]]] = []
    issues: List[ConversionIssue] = []
    used_components: Set[Tuple[int, ...]] = set()

    for variation in parsed.position_variations:
        component = atom_to_fragment.get(variation.source_atom)
        if component is None:
            issues.append(
                _issue(
                    "POSITION_SOURCE_OUT_OF_RANGE",
                    "The position-variation source atom does not exist.",
                    feature=variation.raw,
                    atom_indices=(variation.source_atom,),
                )
            )
            continue
        component_key = tuple(sorted(component))
        if component_key in used_components:
            issues.append(
                _issue(
                    "POSITION_CARRIER_REUSED",
                    "Multiple m: features reuse one carrier fragment.",
                    feature=variation.raw,
                    atom_indices=component_key,
                )
            )
            continue
        used_components.add(component_key)

        if len(component) != 2:
            issues.append(
                _issue(
                    "COMPLEX_POSITION_CARRIER",
                    "Only the explicit two-atom *C position carrier can be "
                    "projected without guessing a substituent abbreviation.",
                    feature=variation.raw,
                    atom_indices=component_key,
                )
            )
            continue
        atoms = [mol.GetAtomWithIdx(index) for index in component]
        if sorted(atom.GetSymbol() for atom in atoms) != ["*", "C"]:
            issues.append(
                _issue(
                    "UNSUPPORTED_POSITION_CARRIER",
                    "The two-atom position carrier is not the expected *C pattern.",
                    feature=variation.raw,
                    atom_indices=component_key,
                )
            )
            continue
        wildcard = next(atom for atom in atoms if atom.GetSymbol() == "*")
        if wildcard.GetIdx() not in parsed.labels:
            issues.append(
                _issue(
                    "UNLABELED_POSITION_CARRIER",
                    "The position-variable carrier has no explicit atom label.",
                    feature=variation.raw,
                    atom_indices=component_key,
                )
            )
            continue
        carrier_bonds = [
            bond
            for bond in mol.GetBonds()
            if bond.GetBeginAtomIdx() in component
            and bond.GetEndAtomIdx() in component
        ]
        if len(carrier_bonds) != 1 or carrier_bonds[0].GetBondType() != Chem.BondType.SINGLE:
            issues.append(
                _issue(
                    "UNSUPPORTED_POSITION_CARRIER_BOND",
                    "The *C position carrier must contain exactly one single bond.",
                    feature=variation.raw,
                    atom_indices=component_key,
                )
            )
            continue
        if component.intersection(variation.target_atoms):
            issues.append(
                _issue(
                    "POSITION_TARGET_IN_CARRIER",
                    "A position target is part of its own carrier fragment.",
                    feature=variation.raw,
                    atom_indices=variation.target_atoms,
                )
            )
            continue
        carriers.append(
            (variation, parsed.labels[wildcard.GetIdx()], component)
        )

    return (carriers if not issues else None), issues


def _assign_position_anchors(
    mol: Chem.Mol,
    carriers: Sequence[Tuple[_PositionVariation, str, Set[int]]],
    forbidden_atoms: Set[int],
) -> Optional[List[int]]:
    """Assign distinct, valence-available representative targets."""

    candidates: List[List[int]] = []
    for variation, _, _ in carriers:
        usable: List[int] = []
        for target in dict.fromkeys(variation.target_atoms):
            if target < 0 or target >= mol.GetNumAtoms() or target in forbidden_atoms:
                continue
            atom = mol.GetAtomWithIdx(target)
            if atom.GetTotalNumHs(includeNeighbors=False) > 0:
                usable.append(target)
        if not usable:
            return None
        candidates.append(usable)

    assignment: List[int] = []

    def search(index: int, used: Set[int]) -> bool:
        if index == len(candidates):
            return True
        for target in candidates[index]:
            if target in used:
                continue
            assignment.append(target)
            used.add(target)
            if search(index + 1, used):
                return True
            used.remove(target)
            assignment.pop()
        return False

    return assignment if search(0, set()) else None


def _graph_projection(
    parsed: _ParsedCXSMILES,
    replacements: Mapping[int, str],
    removed_atoms: Set[int],
    position_nodes: Sequence[Tuple[str, int]],
) -> str:
    """Build a review-only graph projection and serialize marker atoms."""

    editable = Chem.RWMol()
    old_to_new: Dict[int, int] = {}
    marker_text: Dict[int, str] = {}
    next_marker = 900001

    def add_marker(text: str) -> int:
        nonlocal next_marker
        atom = Chem.Atom(0)
        atom.SetAtomMapNum(next_marker)
        new_index = editable.AddAtom(atom)
        marker_text[next_marker] = text
        next_marker += 1
        return new_index

    for old_atom in parsed.mol.GetAtoms():
        old_index = old_atom.GetIdx()
        if old_index in removed_atoms:
            continue
        if old_index in replacements:
            old_to_new[old_index] = add_marker(replacements[old_index])
        else:
            old_to_new[old_index] = editable.AddAtom(Chem.Atom(old_atom))

    copied_bonds: List[Tuple[Chem.Bond, Chem.Bond]] = []
    for old_bond in parsed.mol.GetBonds():
        begin = old_bond.GetBeginAtomIdx()
        end = old_bond.GetEndAtomIdx()
        if begin not in old_to_new or end not in old_to_new:
            continue
        new_begin = old_to_new[begin]
        new_end = old_to_new[end]
        editable.AddBond(new_begin, new_end, old_bond.GetBondType())
        new_bond = editable.GetBondBetweenAtoms(new_begin, new_end)
        new_bond.SetBondDir(old_bond.GetBondDir())
        new_bond.SetIsAromatic(old_bond.GetIsAromatic())
        copied_bonds.append((old_bond, new_bond))

    for label, target in position_nodes:
        marker_index = add_marker("[{}$]".format(label))
        editable.AddBond(old_to_new[target], marker_index, Chem.BondType.SINGLE)

    mol = editable.GetMol()
    for old_bond, new_bond in copied_bonds:
        stereo_atoms = list(old_bond.GetStereoAtoms())
        if stereo_atoms and all(atom in old_to_new for atom in stereo_atoms):
            mapped = [old_to_new[atom] for atom in stereo_atoms]
            if mapped[0] != mapped[1]:
                new_bond.SetStereoAtoms(mapped[0], mapped[1])
                new_bond.SetStereo(old_bond.GetStereo())

    with rdBase.BlockLogs():
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        result = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    for atom_map, text in marker_text.items():
        marker = "[*:{}]".format(atom_map)
        if marker not in result:
            raise ValueError("RDKit marker atom {} was not serialized.".format(atom_map))
        result = result.replace(marker, text)
    return result


def convert_cxsmiles(cxsmiles: str) -> ConversionResult:
    """Convert one CXSMILES into the CLIP-OCSR pseudo-SMILES dialect.

    The function is intentionally conservative.  Inspect ``status`` before
    using ``pseudo_smiles`` as a benchmark label:

    * ``success``: every represented CX feature has a deterministic mapping;
    * ``review_required``: a candidate is supplied, but information is lost;
    * ``unsupported``: no candidate is returned because guessing is required;
    * ``invalid``: the input or its atom indexing cannot be parsed safely.
    """

    parsed, parse_issues = _parse_cxsmiles(cxsmiles)
    if parsed is None:
        issue = parse_issues[0] if parse_issues else _issue(
            "UNKNOWN_PARSE_ERROR", "CXSMILES parsing failed."
        )
        return _invalid_result(cxsmiles, issue)

    metadata = _feature_metadata(parsed)
    errors: List[ConversionIssue] = list(parse_issues)
    warnings: List[ConversionIssue] = []
    atom_count = parsed.mol.GetNumAtoms()

    if parsed.unknown_extension_text:
        errors.append(
            _issue(
                "UNSUPPORTED_CX_EXTENSION",
                "Unrecognized or non-projectable CXSMILES extension text: {}".format(
                    parsed.unknown_extension_text
                ),
                feature=parsed.unknown_extension_text,
            )
        )

    occupied_frequency_atoms: Set[int] = set()
    frequency_replacements: Dict[int, str] = {}
    for group in parsed.frequency_groups:
        invalid_indices = [
            index
            for index in group.atom_indices
            if index < 0 or index >= atom_count
        ]
        if not group.atom_indices or invalid_indices:
            errors.append(
                _issue(
                    "INVALID_FREQUENCY_ATOM_INDEX",
                    "A frequency S-group contains an empty or out-of-range atom list.",
                    feature=group.raw,
                    atom_indices=group.atom_indices,
                )
            )
            continue
        overlap = occupied_frequency_atoms.intersection(group.atom_indices)
        if overlap:
            errors.append(
                _issue(
                    "OVERLAPPING_FREQUENCY_GROUPS",
                    "Overlapping polymer S-groups have no deterministic pseudo-SMILES projection.",
                    feature=group.raw,
                    atom_indices=overlap,
                )
            )
            continue
        occupied_frequency_atoms.update(group.atom_indices)
        if group.group_type != "n":
            errors.append(
                _issue(
                    "UNSUPPORTED_SGROUP_TYPE",
                    "Only SRU (Sg:n) frequency groups are supported.",
                    feature=group.raw,
                    atom_indices=group.atom_indices,
                )
            )
            continue
        if not group.subscript:
            errors.append(
                _issue(
                    "MISSING_FREQUENCY_SUBSCRIPT",
                    "The repeat subscript is empty, so the pseudo atom cannot be named.",
                    feature=group.raw,
                    atom_indices=group.atom_indices,
                )
            )
            continue
        if group.connectivity.lower() not in {"", "ht"}:
            errors.append(
                _issue(
                    "UNSUPPORTED_FREQUENCY_CONNECTIVITY",
                    "Only empty or head-to-tail (ht) S-group connectivity is supported.",
                    feature=group.raw,
                    atom_indices=group.atom_indices,
                )
            )
            continue
        if group.additional_data:
            errors.append(
                _issue(
                    "FREQUENCY_GEOMETRY_UNSUPPORTED",
                    "Additional S-group geometry/data fields are not represented "
                    "by pseudo-SMILES.",
                    feature=group.raw,
                    atom_indices=group.atom_indices,
                )
            )
            continue
        if len(group.atom_indices) != 1:
            errors.append(
                _issue(
                    "MULTI_ATOM_FREQUENCY_GROUP",
                    "A multi-atom repeat has no unique pseudo-SMILES spelling; "
                    "manual transcription is required.",
                    feature=group.raw,
                    atom_indices=group.atom_indices,
                )
            )
            continue

        atom_index = group.atom_indices[0]
        external_bonds = _external_bond_count(parsed.mol, {atom_index})
        if external_bonds > 2:
            errors.append(
                _issue(
                    "BRANCHED_FREQUENCY_GROUP",
                    "A repeat atom with more than two external bonds cannot be "
                    "represented by one linear pseudo atom.",
                    feature=group.raw,
                    atom_indices=(atom_index,),
                )
            )
            continue
        unit, unit_issue = _frequency_unit_label(
            parsed.mol, atom_index, parsed.labels, external_bonds
        )
        if unit_issue is not None:
            errors.append(unit_issue)
            continue
        frequency_replacements[atom_index] = "[({}){}]".format(
            unit, group.subscript
        )
        if external_bonds != 2:
            warnings.append(
                _issue(
                    "TERMINAL_FREQUENCY_GROUP",
                    "The repeat unit has {} external bond(s); verify the displayed "
                    "hydrogen/attachment semantics manually.".format(external_bonds),
                    severity="warning",
                    feature=group.raw,
                    atom_indices=(atom_index,),
                )
            )
        if group.head_bonds or group.tail_bonds:
            warnings.append(
                _issue(
                    "FREQUENCY_CROSSING_BONDS_NOT_ENCODED",
                    "Pseudo-SMILES does not retain explicit S-group crossing-bond indexes.",
                    severity="warning",
                    feature=group.raw,
                    atom_indices=(atom_index,),
                )
            )

    for variation in parsed.position_variations:
        all_indices = (variation.source_atom,) + variation.target_atoms
        if any(index < 0 or index >= atom_count for index in all_indices):
            errors.append(
                _issue(
                    "INVALID_POSITION_ATOM_INDEX",
                    "A position-variation atom index is outside the parsed molecule.",
                    feature=variation.raw,
                    atom_indices=all_indices,
                )
            )
        if occupied_frequency_atoms.intersection(all_indices):
            errors.append(
                _issue(
                    "OVERLAPPING_MARKUSH_FEATURES",
                    "An atom participates in both frequency and position variation; "
                    "the combined notation is not defined in pseudo-SMILES.",
                    feature=variation.raw,
                    atom_indices=occupied_frequency_atoms.intersection(all_indices),
                )
            )

    if errors:
        return _unsupported_result(parsed.source, errors + warnings, metadata)

    replacements: Dict[int, str] = {
        index: "[{}]".format(label) for index, label in parsed.labels.items()
    }
    replacements.update(frequency_replacements)

    if not parsed.position_variations:
        uncovered_wildcards = [
            atom.GetIdx()
            for atom in parsed.mol.GetAtoms()
            if atom.GetSymbol() == "*" and atom.GetIdx() not in replacements
        ]
        if uncovered_wildcards:
            return _unsupported_result(
                parsed.source,
                [
                    _issue(
                        "UNLABELED_WILDCARD",
                        "An unlabeled wildcard remains after applying all supported features.",
                        atom_indices=uncovered_wildcards,
                    )
                ]
                + warnings,
                metadata,
            )
        pseudo = _replace_atom_spans(
            parsed.core, parsed.atom_spans, replacements
        )
        status = (
            ConversionStatus.REVIEW_REQUIRED
            if warnings
            else ConversionStatus.SUCCESS
        )
        return ConversionResult(
            input_cxsmiles=parsed.source,
            status=status,
            pseudo_smiles=pseudo,
            issues=tuple(warnings),
            metadata=metadata,
        )

    carriers, carrier_issues = _position_carriers(parsed)
    if carriers is None:
        return _unsupported_result(
            parsed.source, carrier_issues + warnings, metadata
        )
    removed_atoms = set().union(*(component for _, _, component in carriers))
    uncovered_wildcards = [
        atom.GetIdx()
        for atom in parsed.mol.GetAtoms()
        if atom.GetSymbol() == "*"
        and atom.GetIdx() not in replacements
        and atom.GetIdx() not in removed_atoms
    ]
    if uncovered_wildcards:
        return _unsupported_result(
            parsed.source,
            [
                _issue(
                    "UNLABELED_WILDCARD",
                    "An unlabeled wildcard remains outside the recognized position carriers.",
                    atom_indices=uncovered_wildcards,
                )
            ]
            + warnings,
            metadata,
        )

    anchors = _assign_position_anchors(
        parsed.mol, carriers, removed_atoms.union(occupied_frequency_atoms)
    )
    if anchors is None:
        return _unsupported_result(
            parsed.source,
            [
                _issue(
                    "POSITION_TARGET_ASSIGNMENT_FAILED",
                    "No distinct valence-available representative targets could be assigned.",
                )
            ]
            + warnings,
            metadata,
        )

    # Labels inside removed carrier fragments are represented by the new $ nodes.
    graph_replacements = {
        index: text
        for index, text in replacements.items()
        if index not in removed_atoms
    }
    position_nodes = [
        (carrier[1], anchor) for carrier, anchor in zip(carriers, anchors)
    ]
    try:
        pseudo = _graph_projection(
            parsed,
            graph_replacements,
            removed_atoms,
            position_nodes,
        )
    except Exception as error:  # RDKit reports several exception subclasses.
        return _unsupported_result(
            parsed.source,
            [
                _issue(
                    "POSITION_GRAPH_PROJECTION_FAILED",
                    "RDKit could not construct the review candidate: {}".format(error),
                )
            ]
            + warnings,
            metadata,
        )

    for variation, _, _ in carriers:
        warnings.append(
            _issue(
                "POSITION_TARGET_SET_NOT_ENCODED",
                "The candidate uses one representative attachment atom, but the "
                "pseudo-SMILES $ marker does not retain CXSMILES targets {}.".format(
                    ".".join(str(value) for value in variation.target_atoms)
                ),
                severity="warning",
                feature=variation.raw,
                atom_indices=variation.target_atoms,
            )
        )
    metadata = dict(metadata)
    metadata["representative_position_targets"] = anchors
    return ConversionResult(
        input_cxsmiles=parsed.source,
        status=ConversionStatus.REVIEW_REQUIRED,
        pseudo_smiles=pseudo,
        issues=tuple(warnings),
        metadata=metadata,
    )


def cxsmiles_to_pseudo_smiles(
    cxsmiles: str,
    allow_review: bool = False,
) -> str:
    """Return pseudo-SMILES or raise on a non-exact conversion.

    Set ``allow_review=True`` only when the caller explicitly accepts a
    review-only candidate (currently relevant to projected position variation).
    Benchmark ground truth should normally use the default strict behavior.
    """

    result = convert_cxsmiles(cxsmiles)
    if result.status is ConversionStatus.SUCCESS:
        return result.pseudo_smiles or ""
    if (
        allow_review
        and result.status is ConversionStatus.REVIEW_REQUIRED
        and result.pseudo_smiles is not None
    ):
        return result.pseudo_smiles
    raise CXSMILESToPseudoSMILESError(result)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively convert one CXSMILES to the CLIP-OCSR "
            "pseudo-SMILES dialect."
        )
    )
    parser.add_argument(
        "cxsmiles",
        nargs="?",
        help="CXSMILES string. If omitted, one line is read from standard input.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete machine-readable ConversionResult.",
    )
    parser.add_argument(
        "--allow-review",
        action="store_true",
        help="Accept and print a review_required candidate (exit status 0).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    source = args.cxsmiles
    if source is None:
        source = sys.stdin.readline().strip()
    if not source:
        parser.error("provide CXSMILES as an argument or one line on standard input")

    result = convert_cxsmiles(source)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    elif result.pseudo_smiles is not None:
        print(result.pseudo_smiles)

    if result.status is ConversionStatus.SUCCESS:
        return 0
    if result.status is ConversionStatus.REVIEW_REQUIRED:
        message = (
            "Review required: the candidate pseudo-SMILES does not retain all "
            "CXSMILES semantics; please inspect it manually."
        )
        print(message, file=sys.stderr)
        for issue in result.issues:
            print("- {}: {}".format(issue.code, issue.message), file=sys.stderr)
        return 0 if args.allow_review else 3

    print(
        "This CXSMILES cannot be converted to pseudo-SMILES automatically; "
        "please inspect it manually.",
        file=sys.stderr,
    )
    for issue in result.issues:
        print("- {}: {}".format(issue.code, issue.message), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
