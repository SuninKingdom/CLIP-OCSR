#!/usr/bin/env python3
"""Convert OCSR-predicted V2000 MOL files to pseudo-SMILES.

The default ``experiment`` mode reproduces the conversion logic used for the
MolScribe, MolNexTR, and ChemEAGLE comparisons in this work.  A more defensive
``strict`` mode is retained for exploratory use.  Every batch input row is
retained; missing, unsupported, and invalid predictions receive an empty
pseudo-SMILES and an explicit diagnostic status.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem, rdBase


CONVERTER_VERSION = "2.0.0"
EXPERIMENT_RDKIT_VERSION = "2022.09.1"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PERIODIC_ELEMENT_SYMBOLS = {
    Chem.GetPeriodicTable().GetElementSymbol(atomic_number)
    for atomic_number in range(1, 119)
}
ELEMENT_SYMBOLS = PERIODIC_ELEMENT_SYMBOLS | {"D", "T"}
ALIAS_LINE_PATTERN = re.compile(r"^\s*A\s+(\d+)\s*$")


class ConversionStatus(str, Enum):
    """Outcome of one MOL-file conversion."""

    SUCCESS = "success"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    MISSING_INPUT = "missing_input"


class ConversionMode(str, Enum):
    """Available MOL-to-pseudo-SMILES conversion algorithms."""

    EXPERIMENT = "experiment"
    STRICT = "strict"


@dataclass(frozen=True)
class ConversionIssue:
    """Machine-readable conversion diagnostic."""

    code: str
    message: str
    atom_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["atom_indices"] = list(self.atom_indices)
        return value


@dataclass(frozen=True)
class ConversionResult:
    """Auditable MOL-to-pseudo-SMILES result."""

    source: str | None
    status: ConversionStatus
    pseudo_smiles: str | None = None
    issues: tuple[ConversionIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status.value,
            "pseudo_smiles": self.pseudo_smiles,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": dict(self.metadata),
        }


def _issue(code: str, message: str, *atom_indices: int) -> ConversionIssue:
    return ConversionIssue(code=code, message=message, atom_indices=tuple(atom_indices))


def _failure(
    source: str | None,
    status: ConversionStatus,
    issues: Iterable[ConversionIssue],
    metadata: Mapping[str, Any] | None = None,
) -> ConversionResult:
    return ConversionResult(
        source=source,
        status=status,
        issues=tuple(issues),
        metadata=metadata or {},
    )


def _counts_layout(lines: list[str]) -> tuple[int, int] | ConversionResult:
    for index, line in enumerate(lines):
        if "V3000" in line:
            return _failure(
                None,
                ConversionStatus.UNSUPPORTED,
                [
                    _issue(
                        "UNSUPPORTED_MOLFILE_VERSION",
                        "V3000 input is not projected by this benchmark converter; use V2000.",
                    )
                ],
            )
        if "V2000" not in line:
            continue
        try:
            atom_count = int(line[:3])
        except ValueError:
            try:
                atom_count = int(line.split()[0])
            except (IndexError, ValueError):
                return _failure(
                    None,
                    ConversionStatus.INVALID,
                    [_issue("INVALID_COUNTS_LINE", "Cannot read the V2000 atom count.")],
                )
        if atom_count < 0 or index + 1 + atom_count > len(lines):
            return _failure(
                None,
                ConversionStatus.INVALID,
                [_issue("TRUNCATED_ATOM_BLOCK", "The V2000 atom block is incomplete.")],
            )
        return index, atom_count
    return _failure(
        None,
        ConversionStatus.INVALID,
        [_issue("MISSING_COUNTS_LINE", "No V2000 or V3000 counts line was found.")],
    )


def _atom_symbol(line: str) -> str:
    if len(line) >= 34:
        return line[31:34].strip()
    fields = line.split()
    return fields[3] if len(fields) > 3 else ""


def _replace_atom_symbol(line: str, symbol: str = "*") -> str:
    if len(line) < 34:
        raise ValueError("V2000 atom lines must contain the fixed-width symbol field")
    return line[:31] + f"{symbol:<3}" + line[34:]


def _normalize_label(value: str) -> str:
    label = value.strip()
    if label.startswith("[") and label.endswith("]"):
        label = label[1:-1].strip()
    if label == "*":
        return "R"
    numbered_star = re.fullmatch(r"(\d+)\*", label)
    if numbered_star:
        return "R" + numbered_star.group(1)
    if not label:
        raise ValueError("empty pseudo-atom label")
    if any(character in label for character in "[]\r\n:"):
        raise ValueError(f"unsupported pseudo-atom label {value!r}")
    if any(character.isspace() for character in label):
        raise ValueError(f"pseudo-atom labels cannot contain whitespace: {value!r}")
    return label


def _alias_records(
    lines: list[str], start: int, atom_count: int
) -> tuple[dict[int, str], list[ConversionIssue]]:
    aliases: dict[int, str] = {}
    issues: list[ConversionIssue] = []
    for line_index in range(start, len(lines)):
        match = ALIAS_LINE_PATTERN.match(lines[line_index])
        if match is None:
            continue
        atom_number = int(match.group(1))
        if atom_number < 1 or atom_number > atom_count:
            issues.append(
                _issue(
                    "ALIAS_ATOM_INDEX_OUT_OF_RANGE",
                    f"Alias record refers to atom {atom_number}, outside 1..{atom_count}.",
                    atom_number,
                )
            )
            continue
        if line_index + 1 >= len(lines):
            issues.append(
                _issue(
                    "TRUNCATED_ALIAS_RECORD",
                    f"Alias record for atom {atom_number} has no label line.",
                    atom_number,
                )
            )
            continue
        try:
            label = _normalize_label(lines[line_index + 1])
        except ValueError as error:
            issues.append(_issue("INVALID_ALIAS_LABEL", str(error), atom_number))
            continue
        previous = aliases.get(atom_number - 1)
        if previous is not None and previous != label:
            issues.append(
                _issue(
                    "CONFLICTING_ALIAS_LABELS",
                    f"Atom {atom_number} has conflicting aliases {previous!r} and {label!r}.",
                    atom_number,
                )
            )
            continue
        aliases[atom_number - 1] = label
    return aliases, issues


def _rgroup_records(
    lines: list[str], atom_count: int
) -> tuple[dict[int, str], list[ConversionIssue]]:
    labels: dict[int, str] = {}
    issues: list[ConversionIssue] = []
    for line in lines:
        if not line.startswith("M  RGP"):
            continue
        fields = line.split()
        try:
            pair_count = int(fields[2])
            values = [int(value) for value in fields[3:]]
        except (IndexError, ValueError):
            issues.append(_issue("INVALID_RGP_RECORD", f"Cannot parse R-group line: {line}"))
            continue
        if len(values) < pair_count * 2:
            issues.append(_issue("TRUNCATED_RGP_RECORD", f"Incomplete R-group line: {line}"))
            continue
        for offset in range(pair_count):
            atom_number, group_number = values[offset * 2 : offset * 2 + 2]
            if atom_number < 1 or atom_number > atom_count or group_number < 0:
                issues.append(
                    _issue(
                        "INVALID_RGP_VALUE",
                        f"Invalid atom/group pair ({atom_number}, {group_number}).",
                        atom_number,
                    )
                )
                continue
            atom_index = atom_number - 1
            label = f"R{group_number}"
            previous = labels.get(atom_index)
            if previous is not None and previous != label:
                issues.append(
                    _issue(
                        "CONFLICTING_RGP_LABELS",
                        f"Atom {atom_number} has conflicting R-group numbers.",
                        atom_number,
                    )
                )
            else:
                labels[atom_index] = label
    return labels, issues


def _merge_label(
    labels: dict[int, str], atom_index: int, candidate: str, issues: list[ConversionIssue]
) -> None:
    previous = labels.get(atom_index)
    if previous is not None and previous != candidate:
        issues.append(
            _issue(
                "CONFLICTING_PSEUDO_LABELS",
                f"Atom {atom_index + 1} is labeled both {previous!r} and {candidate!r}.",
                atom_index + 1,
            )
        )
    else:
        labels[atom_index] = candidate


def _coerce_mode(mode: ConversionMode | str) -> ConversionMode:
    if isinstance(mode, ConversionMode):
        return mode
    try:
        return ConversionMode(mode)
    except ValueError as error:
        choices = ", ".join(item.value for item in ConversionMode)
        raise ValueError(f"Unknown conversion mode {mode!r}; choose one of: {choices}") from error


def _mode_metadata(mode: ConversionMode) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "converter_version": CONVERTER_VERSION,
        "conversion_mode": mode.value,
        "rdkit_version": rdBase.rdkitVersion,
    }
    if mode is ConversionMode.EXPERIMENT:
        metadata.update(
            {
                "experiment_rdkit_version": EXPERIMENT_RDKIT_VERSION,
                "rdkit_version_matches_experiment": (
                    rdBase.rdkitVersion == EXPERIMENT_RDKIT_VERSION
                ),
                "conversion_algorithm": "trans2smilesR_molscribe",
            }
        )
    return metadata


def _annotate_result(
    result: ConversionResult, mode: ConversionMode
) -> ConversionResult:
    metadata = _mode_metadata(mode)
    metadata.update(result.metadata)
    return ConversionResult(
        source=result.source,
        status=result.status,
        pseudo_smiles=result.pseudo_smiles,
        issues=result.issues,
        metadata=metadata,
    )


def _experiment_alias_label(label: str) -> str:
    """Apply the alias normalization used in the reported experiments."""

    if label == "*":
        return "R"
    if "*" in label and label.endswith("*"):
        return "R" + label.replace("*", "")
    return label


def _convert_molblock_experiment(
    mol_block: str, source: str | None = None
) -> ConversionResult:
    """Reproduce the MOL conversion used to obtain the reported baselines.

    This is a structured port of ``trans2smilesR_molscribe.py``.  Its order of
    operations is intentionally preserved: RDKit first reads the unmodified
    prediction, aliases and other non-element atom symbols are replaced by
    temporary ``R70xx`` atoms, RDKit serializes the modified graph, and the
    original labels are restored in the resulting SMILES string.
    """

    metadata = _mode_metadata(ConversionMode.EXPERIMENT)
    if not isinstance(mol_block, str) or not mol_block.strip():
        return _failure(
            source,
            ConversionStatus.INVALID,
            [_issue("EMPTY_MOL_BLOCK", "The MOL block is empty.")],
            metadata,
        )

    # The historical batch scripts called Chem.MolFromMolFile first and did
    # not write a prediction row when this initial parse failed.  Reading the
    # same text with MolFromMolBlock has the equivalent RDKit behavior while
    # allowing the public converter to retain an auditable failure row.
    try:
        with rdBase.BlockLogs():
            initial_mol = Chem.MolFromMolBlock(mol_block)
    except Exception as error:
        return _failure(
            source,
            ConversionStatus.INVALID,
            [_issue("EXPERIMENT_INITIAL_PARSE_FAILED", str(error))],
            metadata,
        )
    if initial_mol is None:
        return _failure(
            source,
            ConversionStatus.INVALID,
            [
                _issue(
                    "EXPERIMENT_INITIAL_PARSE_FAILED",
                    "RDKit rejected the unmodified MOL prediction, as in the "
                    "reported conversion workflow.",
                )
            ],
            metadata,
        )

    atom_count = initial_mol.GetNumAtoms()
    metadata.update({"atom_count": atom_count})
    lines = mol_block.split("\n")
    alias_indices: list[int] = []
    substituents: list[str] = []
    for line_index, line in enumerate(lines):
        if not line.startswith("A  "):
            continue
        fields = line.split()
        if len(fields) < 2 or line_index + 1 >= len(lines):
            return _failure(
                source,
                ConversionStatus.INVALID,
                [_issue("INVALID_ALIAS_RECORD", f"Cannot parse alias record: {line}")],
                metadata,
            )
        try:
            atom_number = int(fields[1])
        except ValueError:
            return _failure(
                source,
                ConversionStatus.INVALID,
                [_issue("INVALID_ALIAS_RECORD", f"Cannot parse alias record: {line}")],
                metadata,
            )
        alias_indices.append(atom_number)
        substituents.append(_experiment_alias_label(lines[line_index + 1]))

    temporary_labels: list[str] = []
    for offset, atom_number in enumerate(alias_indices, start=1):
        temporary_label = f"R{7000 + offset}"
        temporary_labels.append(temporary_label)
        atom_line_index = atom_number + 4 - 1
        if atom_line_index >= len(lines):
            return _failure(
                source,
                ConversionStatus.INVALID,
                [
                    _issue(
                        "ALIAS_ATOM_INDEX_OUT_OF_RANGE",
                        f"Alias record refers to unavailable atom {atom_number}.",
                        atom_number,
                    )
                ],
                metadata,
            )
        fields = lines[atom_line_index].split()
        if len(fields) > 3:
            original_symbol = fields[3]
            lines[atom_line_index] = lines[atom_line_index].replace(
                original_symbol, temporary_label
            )

    # Preserve the historical fixed V2000 layout assumption (four header
    # lines) and the original periodic-element allowlist.
    for atom_line_index in range(4, 4 + atom_count):
        if atom_line_index >= len(lines):
            return _failure(
                source,
                ConversionStatus.INVALID,
                [_issue("TRUNCATED_ATOM_BLOCK", "The MOL atom block is incomplete.")],
                metadata,
            )
        fields = lines[atom_line_index].split()
        if len(fields) <= 3:
            continue
        symbol = fields[3]
        if symbol in PERIODIC_ELEMENT_SYMBOLS or symbol.startswith("R70"):
            continue
        atom_number = atom_line_index - 4 + 1
        alias_indices.append(atom_number)
        substituents.append(symbol)
        temporary_label = f"R{7000 + len(temporary_labels) + 1}"
        temporary_labels.append(temporary_label)
        lines[atom_line_index] = lines[atom_line_index].replace(
            symbol, temporary_label
        )

    rewritten_block = "\n".join(lines)
    try:
        with rdBase.BlockLogs():
            rewritten_mol = Chem.MolFromMolBlock(rewritten_block)
        if rewritten_mol is None:
            raise ValueError("RDKit could not parse the placeholder-rewritten MOL block")
        pseudo_smiles = Chem.MolToSmiles(rewritten_mol)
    except Exception as error:
        return _failure(
            source,
            ConversionStatus.INVALID,
            [_issue("EXPERIMENT_PLACEHOLDER_PARSE_FAILED", str(error))],
            metadata,
        )

    for index in reversed(range(len(substituents))):
        pseudo_smiles = pseudo_smiles.replace(
            f"{index + 1}*:0", substituents[index]
        )
    metadata["pseudo_atom_count"] = len(substituents)
    if "*" in pseudo_smiles:
        return _failure(
            source,
            ConversionStatus.UNSUPPORTED,
            [
                _issue(
                    "UNRESOLVED_WILDCARD",
                    "A wildcard remains after the experiment-compatible label restoration.",
                )
            ],
            metadata,
        )
    return ConversionResult(
        source=source,
        status=ConversionStatus.SUCCESS,
        pseudo_smiles=pseudo_smiles,
        metadata=metadata,
    )


def _convert_molblock_strict(
    mol_block: str, source: str | None = None
) -> ConversionResult:
    """Defensively convert one V2000 MOL block to pseudo-SMILES."""

    if not isinstance(mol_block, str) or not mol_block.strip():
        return _failure(
            source,
            ConversionStatus.INVALID,
            [_issue("EMPTY_MOL_BLOCK", "The MOL block is empty.")],
        )
    lines = mol_block.splitlines()
    layout = _counts_layout(lines)
    if isinstance(layout, ConversionResult):
        return ConversionResult(
            source=source,
            status=layout.status,
            issues=layout.issues,
            metadata=layout.metadata,
        )
    counts_index, atom_count = layout
    atom_start = counts_index + 1
    atom_end = atom_start + atom_count
    raw_symbols = [_atom_symbol(line) for line in lines[atom_start:atom_end]]
    if any(not symbol for symbol in raw_symbols):
        bad_atoms = tuple(index + 1 for index, value in enumerate(raw_symbols) if not value)
        return _failure(
            source,
            ConversionStatus.INVALID,
            [_issue("INVALID_ATOM_LINE", "One or more atom symbols are missing.", *bad_atoms)],
        )

    aliases, issues = _alias_records(lines, atom_end, atom_count)
    rgroups, rgroup_issues = _rgroup_records(lines, atom_count)
    issues.extend(rgroup_issues)
    labels = dict(aliases)
    for atom_index, label in rgroups.items():
        _merge_label(labels, atom_index, label, issues)

    unresolved_raw_dummies: set[int] = set()
    for atom_index, symbol in enumerate(raw_symbols):
        if symbol in ELEMENT_SYMBOLS:
            continue
        # An explicit alias or M  RGP record is more specific than the raw
        # carrier symbol (commonly ``R``, ``R#``, or ``*``).
        if atom_index in labels:
            continue
        if symbol == "*":
            unresolved_raw_dummies.add(atom_index)
            continue
        if symbol == "R#":
            unresolved_raw_dummies.add(atom_index)
            continue
        try:
            label = _normalize_label(symbol)
        except ValueError as error:
            issues.append(_issue("INVALID_ATOM_LABEL", str(error), atom_index + 1))
            continue
        _merge_label(labels, atom_index, label, issues)

    if issues:
        return _failure(source, ConversionStatus.UNSUPPORTED, issues)

    rewritten = list(lines)
    for atom_index, symbol in enumerate(raw_symbols):
        if atom_index in labels or symbol not in ELEMENT_SYMBOLS:
            try:
                rewritten[atom_start + atom_index] = _replace_atom_symbol(
                    rewritten[atom_start + atom_index]
                )
            except ValueError as error:
                return _failure(
                    source,
                    ConversionStatus.INVALID,
                    [_issue("INVALID_ATOM_LINE", str(error), atom_index + 1)],
                )

    rewritten_block = "\n".join(rewritten) + ("\n" if mol_block.endswith("\n") else "")
    with rdBase.BlockLogs():
        mol = Chem.MolFromMolBlock(
            rewritten_block,
            sanitize=False,
            removeHs=False,
            strictParsing=False,
        )
    if mol is None or mol.GetNumAtoms() != atom_count:
        return _failure(
            source,
            ConversionStatus.INVALID,
            [_issue("RDKIT_PARSE_FAILED", "RDKit could not parse the rewritten MOL block.")],
        )

    for atom_index in sorted(unresolved_raw_dummies):
        atom = mol.GetAtomWithIdx(atom_index)
        candidate: str | None = None
        if atom.HasProp("molFileAlias"):
            try:
                candidate = _normalize_label(atom.GetProp("molFileAlias"))
            except ValueError:
                candidate = None
        if candidate is None and atom.HasProp("_MolFileRLabel"):
            value = atom.GetProp("_MolFileRLabel").strip()
            if value.isdigit():
                candidate = "R" + value
        if candidate is None and atom.HasProp("dummyLabel"):
            value = atom.GetProp("dummyLabel").strip()
            if value not in {"", "*", "R#"}:
                try:
                    candidate = _normalize_label(value)
                except ValueError:
                    candidate = None
        if candidate is not None:
            labels[atom_index] = candidate

    unresolved = [
        atom.GetIdx() + 1
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetIdx() not in labels
    ]
    if unresolved:
        return _failure(
            source,
            ConversionStatus.UNSUPPORTED,
            [
                _issue(
                    "UNLABELED_DUMMY_ATOM",
                    "A wildcard/query atom has no alias or R-group number, so "
                    "its pseudo label cannot be inferred.",
                    *unresolved,
                )
            ],
            {"molfile_version": "V2000", "atom_count": atom_count},
        )

    removed_atom_maps = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            removed_atom_maps += 1
            atom.SetAtomMapNum(0)

    marker_labels: dict[int, str] = {}
    next_marker = 900_000
    for atom_index, label in sorted(labels.items()):
        atom = mol.GetAtomWithIdx(atom_index)
        atom.SetAtomicNum(0)
        atom.SetIsotope(0)
        next_marker += 1
        atom.SetAtomMapNum(next_marker)
        marker_labels[next_marker] = label

    try:
        with rdBase.BlockLogs():
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            pseudo_smiles = Chem.MolToSmiles(
                mol, canonical=True, isomericSmiles=True
            )
    except Exception as error:
        return _failure(
            source,
            ConversionStatus.INVALID,
            [_issue("RDKIT_SANITIZATION_FAILED", f"RDKit rejected the molecular graph: {error}")],
            {"molfile_version": "V2000", "atom_count": atom_count},
        )

    for marker, label in marker_labels.items():
        serialized = f"[*:{marker}]"
        if serialized not in pseudo_smiles:
            return _failure(
                source,
                ConversionStatus.INVALID,
                [
                    _issue(
                        "PSEUDO_MARKER_SERIALIZATION_FAILED",
                        f"RDKit did not serialize temporary marker {marker} as expected.",
                    )
                ],
            )
        pseudo_smiles = pseudo_smiles.replace(serialized, f"[{label}]")

    if re.search(r"\*", pseudo_smiles):
        return _failure(
            source,
            ConversionStatus.UNSUPPORTED,
            [_issue("UNRESOLVED_WILDCARD", "A wildcard remains after label restoration.")],
        )
    return ConversionResult(
        source=source,
        status=ConversionStatus.SUCCESS,
        pseudo_smiles=pseudo_smiles,
        metadata={
            "converter_version": CONVERTER_VERSION,
            "molfile_version": "V2000",
            "atom_count": atom_count,
            "pseudo_atom_count": len(labels),
            "removed_atom_map_count": removed_atom_maps,
        },
    )


def convert_molblock(
    mol_block: str,
    source: str | None = None,
    mode: ConversionMode | str = ConversionMode.EXPERIMENT,
) -> ConversionResult:
    """Convert one MOL block with the selected conversion algorithm.

    ``experiment`` is the default because it is the algorithm used for the
    comparison results reported in this work.  ``strict`` is an optional,
    more defensive projection and must not be substituted silently when
    reproducing the reported benchmark values.
    """

    selected_mode = _coerce_mode(mode)
    if selected_mode is ConversionMode.EXPERIMENT:
        result = _convert_molblock_experiment(mol_block, source=source)
    else:
        result = _convert_molblock_strict(mol_block, source=source)
    return _annotate_result(result, selected_mode)


def convert_molfile(
    path: str | Path,
    mode: ConversionMode | str = ConversionMode.EXPERIMENT,
) -> ConversionResult:
    """Read and convert one MOL file."""

    selected_mode = _coerce_mode(mode)
    input_path = Path(path)
    source = str(input_path)
    if not input_path.is_file():
        return _annotate_result(
            _failure(
                source,
                ConversionStatus.MISSING_INPUT,
                [_issue("MOL_FILE_NOT_FOUND", f"MOL file not found: {input_path}")],
            ),
            selected_mode,
        )
    try:
        mol_block = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            mol_block = input_path.read_text(encoding="latin-1")
        except OSError as error:
            return _annotate_result(
                _failure(
                    source,
                    ConversionStatus.INVALID,
                    [_issue("MOL_FILE_READ_FAILED", str(error))],
                ),
                selected_mode,
            )
    except OSError as error:
        return _annotate_result(
            _failure(
                source,
                ConversionStatus.INVALID,
                [_issue("MOL_FILE_READ_FAILED", str(error))],
            ),
            selected_mode,
        )
    return convert_molblock(mol_block, source=source, mode=selected_mode)


def _image_name(value: Any, suffix: str) -> str:
    name = str(value).strip()
    if not name or not suffix or Path(name).suffix.lower() in IMAGE_SUFFIXES:
        return name
    return name + suffix


def _serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _fieldnames(records: Iterable[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for record in records:
        for name in record:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _write_csv(path: Path, records: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = _fieldnames(records)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(path.parent),
            prefix=f".{path.name}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
            writer.writeheader()
            for record in records:
                writer.writerow({name: _serialize_cell(record.get(name)) for name in names})
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def _append_result(
    record: dict[str, Any], result: ConversionResult, pseudo_column: str
) -> None:
    value = result.to_dict()
    record[pseudo_column] = result.pseudo_smiles or ""
    record["conversion_status"] = result.status.value
    record["conversion_issue_codes"] = ";".join(
        issue["code"] for issue in value["issues"]
    )
    record["conversion_issues"] = value["issues"]
    record["conversion_metadata"] = value["metadata"]


def _directory_records(
    directory: Path,
    recursive: bool,
    image_column: str,
    image_suffix: str,
    pseudo_column: str,
    mode: ConversionMode,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    mol_paths = sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() == ".mol"
    )
    if not mol_paths:
        raise ValueError(f"No MOL files found in {directory}")
    records: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    seen_images: set[str] = set()
    for mol_path in mol_paths:
        image_name = _image_name(mol_path.stem, image_suffix)
        if image_name in seen_images:
            raise ValueError(f"Duplicate derived image name: {image_name}")
        seen_images.add(image_name)
        result = convert_molfile(mol_path, mode=mode)
        record: dict[str, Any] = {
            image_column: image_name,
            "mol_file": str(mol_path),
        }
        _append_result(record, result, pseudo_column)
        records.append(record)
        statuses[result.status.value] += 1
    return records, statuses


def _manifest_records(
    manifest: Path,
    mol_column: str,
    source_image_column: str | None,
    image_column: str,
    image_suffix: str,
    pseudo_column: str,
    mode: ConversionMode,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        records = [dict(row) for row in csv.DictReader(handle)]
    if not records:
        raise ValueError(f"No records found in {manifest}")
    if mol_column not in records[0]:
        raise ValueError(f"MOL path column {mol_column!r} is missing from {manifest}")
    if source_image_column is None:
        source_image_column = next(
            (
                name
                for name in ("image_name", "Image_Name", "id")
                if name in records[0]
            ),
            None,
        )

    statuses: Counter[str] = Counter()
    for row_number, record in enumerate(records, start=1):
        value = str(record.get(mol_column, "")).strip()
        mol_path = Path(value) if value else Path("__missing_mol_file__")
        if value and not mol_path.is_absolute():
            mol_path = manifest.parent / mol_path
        result = (
            convert_molfile(mol_path, mode=mode)
            if value
            else _annotate_result(
                _failure(
                    None,
                    ConversionStatus.MISSING_INPUT,
                    [_issue("MISSING_MOL_PATH", f"Row {row_number} has no MOL path.")],
                ),
                mode,
            )
        )
        if source_image_column and str(record.get(source_image_column, "")).strip():
            base_name = record[source_image_column]
        else:
            base_name = mol_path.stem if value else ""
        record[image_column] = _image_name(base_name, image_suffix)
        _append_result(record, result, pseudo_column)
        statuses[result.status.value] += 1
    return records, statuses


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert V2000 MOL predictions to auditable pseudo-SMILES."
    )
    parser.add_argument(
        "input", type=Path, help="One MOL file, a MOL directory, or a CSV manifest"
    )
    parser.add_argument(
        "--output", type=Path, help="Required CSV output for a directory or manifest"
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan a MOL directory")
    parser.add_argument(
        "--mol-column",
        default="mol_file",
        help="MOL path column in a CSV manifest",
    )
    parser.add_argument("--source-image-column", help="Existing image/id column in a CSV manifest")
    parser.add_argument("--image-column", default="Image_Name")
    parser.add_argument("--image-suffix", default=".png")
    parser.add_argument("--pseudo-column", default="Predicted_SMILES")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ConversionMode],
        default=ConversionMode.EXPERIMENT.value,
        help=(
            "Conversion algorithm. 'experiment' reproduces the reported "
            "comparison workflow and is the default; 'strict' is optional."
        ),
    )
    parser.add_argument(
        "--allow-rdkit-version-mismatch",
        action="store_true",
        help=(
            "Allow experiment mode to run with an RDKit version other than "
            f"{EXPERIMENT_RDKIT_VERSION}. Results may then differ from the paper."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print a single-file result as JSON")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    mode = ConversionMode(args.mode)
    if mode is ConversionMode.EXPERIMENT and rdBase.rdkitVersion != EXPERIMENT_RDKIT_VERSION:
        message = (
            "Experiment mode requires RDKit "
            f"{EXPERIMENT_RDKIT_VERSION} for exact paper reproduction, but "
            f"this environment provides {rdBase.rdkitVersion}."
        )
        if not args.allow_rdkit_version_mismatch:
            print(
                f"ERROR: {message} Use the documented reproduction environment, "
                "select --mode strict, or explicitly pass "
                "--allow-rdkit-version-mismatch for a non-reproduction run.",
                file=sys.stderr,
            )
            return 2
        print(f"WARNING: {message}", file=sys.stderr)
    input_path = args.input.resolve()
    if input_path.suffix.lower() == ".mol":
        result = convert_molfile(input_path, mode=mode)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        elif result.pseudo_smiles is not None:
            print(result.pseudo_smiles)
        if result.status is not ConversionStatus.SUCCESS:
            for issue in result.issues:
                print(f"- {issue.code}: {issue.message}", file=sys.stderr)
            return 2
        return 0

    if args.output is None:
        raise ValueError("--output is required for a directory or CSV manifest")
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must differ")
    if input_path.is_dir():
        records, statuses = _directory_records(
            input_path,
            args.recursive,
            args.image_column,
            args.image_suffix,
            args.pseudo_column,
            mode,
        )
    elif input_path.is_file() and input_path.suffix.lower() == ".csv":
        records, statuses = _manifest_records(
            input_path,
            args.mol_column,
            args.source_image_column,
            args.image_column,
            args.image_suffix,
            args.pseudo_column,
            mode,
        )
    else:
        raise ValueError("Input must be a MOL file, a directory, or a CSV manifest")
    _write_csv(output_path, records, args.overwrite)
    print(
        f"Converted {len(records)} rows to {output_path}; "
        f"statuses={dict(sorted(statuses.items()))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
