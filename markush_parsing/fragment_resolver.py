"""Resolve Markush variable descriptions to attachable molecular fragments.

The bundled ``resources/markush_fragment_library.json`` file is a
description/name-to-fragment lookup table migrated from an earlier internal
implementation. Keeping the data inside this repository makes the parser
self-contained while retaining the mappings as an optional, auditable source.
This module adds deterministic normalization, curated high-confidence
mappings, RDKit validation, and canonical deduplication.

Fragments use a single mapped dummy atom (``[*:1]``) as the attachment point.
Simple atom replacements and hydrogen removal are represented explicitly so
the downstream instantiator can operate on molecular graphs rather than by
SMILES string slicing.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rdkit import Chem, rdBase


DEFAULT_FRAGMENT_LIBRARY_PATH = (
    Path(__file__).resolve().parent
    / "resources"
    / "markush_fragment_library.json"
)


@dataclass(frozen=True)
class FragmentCandidate:
    """One concrete interpretation of a textual Markush variable value."""

    source_value: str
    name: str
    kind: str
    source: str
    coverage: str
    fragment_smiles: str | None = None
    atom_smiles: str | None = None
    canonical_fragment_smiles: str | None = None
    library_description: str | None = None

    def key(self) -> tuple:
        return (
            self.kind,
            self.atom_smiles,
            self.canonical_fragment_smiles or self.fragment_smiles,
        )

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


def normalize_descriptor(value: str) -> str:
    """Normalize a description conservatively for deterministic lookup."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.translate(str.maketrans({"–": "-", "—": "-", "−": "-"}))
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(?:a|an|the)\s+", "", text)

    # Common prose range forms used in patent variable definitions.
    text = re.sub(
        r"\b(alkyl|alkoxy)\s+(?:group\s+)?(?:having|containing|of)\s+"
        r"(?:c\s*)?(\d+)\s*(?:-|to)\s*(?:c\s*)?(\d+)\s+carbon\s+atoms?\b",
        lambda match: f"c{match.group(2)}-c{match.group(3)} {match.group(1)}",
        text,
    )
    text = re.sub(
        r"\b(alkyl|alkoxy)\s+(?:group\s+)?(?:having|containing|of)\s+"
        r"c\s*(\d+)\s*(?:-|to)\s*c?\s*(\d+)\b",
        lambda match: f"c{match.group(2)}-c{match.group(3)} {match.group(1)}",
        text,
    )
    text = re.sub(
        r"\bc\s*(\d+)\s*(?:-|to)\s*c?\s*(\d+)\b",
        lambda match: f"c{match.group(1)}-c{match.group(2)}",
        text,
    )
    text = re.sub(r"\s+(?:group|atom)$", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    return text


def _candidate(
    *,
    name: str,
    kind: str,
    source: str = "curated_builtin",
    coverage: str = "exact",
    fragment_smiles: str | None = None,
    atom_smiles: str | None = None,
) -> FragmentCandidate:
    return FragmentCandidate(
        source_value="",
        name=name,
        kind=kind,
        source=source,
        coverage=coverage,
        fragment_smiles=fragment_smiles,
        atom_smiles=atom_smiles,
        canonical_fragment_smiles=fragment_smiles,
    )


def _atom(name: str, atom_smiles: str, coverage: str = "exact") -> FragmentCandidate:
    return _candidate(
        name=name,
        kind="atom",
        atom_smiles=atom_smiles,
        coverage=coverage,
    )


def _fragment(
    name: str, fragment_smiles: str, coverage: str = "exact"
) -> FragmentCandidate:
    return _candidate(
        name=name,
        kind="fragment",
        fragment_smiles=fragment_smiles,
        coverage=coverage,
    )


def _builtin_candidates(key: str) -> list[FragmentCandidate]:
    """Return reviewed mappings for common finite or explicit groups."""
    if key in {"hydrogen", "hydro", "h"}:
        return [_candidate(name="hydrogen", kind="hydrogen")]
    if key in {"deuterium", "d"}:
        return [_fragment("deuterium", "[*:1][2H]")]
    if key in {"halogen", "halo"} or re.fullmatch(r"halogen\s*\([^)]*\)", key):
        return [
            _atom("fluorine", "F", "complete_finite_class"),
            _atom("chlorine", "Cl", "complete_finite_class"),
            _atom("bromine", "Br", "complete_finite_class"),
            _atom("iodine", "I", "complete_finite_class"),
        ]

    atom_aliases = {
        "f": ("fluorine", "F"),
        "fluorine": ("fluorine", "F"),
        "fluoro": ("fluorine", "F"),
        "fluoride": ("fluorine", "F"),
        "cl": ("chlorine", "Cl"),
        "chlorine": ("chlorine", "Cl"),
        "chloro": ("chlorine", "Cl"),
        "chloride": ("chlorine", "Cl"),
        "br": ("bromine", "Br"),
        "bromine": ("bromine", "Br"),
        "bromo": ("bromine", "Br"),
        "bromide": ("bromine", "Br"),
        "i": ("iodine", "I"),
        "iodine": ("iodine", "I"),
        "iodo": ("iodine", "I"),
        "iodide": ("iodine", "I"),
        "oxygen": ("oxygen", "O"),
        "o": ("oxygen", "O"),
        "sulfur": ("sulfur", "S"),
        "sulphur": ("sulfur", "S"),
        "s": ("sulfur", "S"),
        "-s-": ("sulfur linker", "S"),
        "nitrogen": ("nitrogen", "N"),
        "n": ("nitrogen", "N"),
        "nh": ("NH linker", "N"),
        "-nh-": ("NH linker", "N"),
        "carbon": ("carbon", "C"),
        "c": ("carbon", "C"),
        "ch": ("methine", "C"),
        "ch2": ("methylene", "C"),
        "-o-": ("oxygen linker", "O"),
    }
    if key in atom_aliases:
        name, atom_smiles = atom_aliases[key]
        return [_atom(name, atom_smiles)]

    fragment_aliases = {
        "methyl": ("methyl", "[*:1]C"),
        "ch3": ("methyl", "[*:1]C"),
        "hydroxy": ("hydroxy", "[*:1]O"),
        "hydroxyl": ("hydroxy", "[*:1]O"),
        "oh": ("hydroxy", "[*:1]O"),
        "amino": ("amino", "[*:1]N"),
        "nh2": ("amino", "[*:1]N"),
        "thiol": ("thiol", "[*:1]S"),
        "mercapto": ("thiol", "[*:1]S"),
        "ethyl": ("ethyl", "[*:1]CC"),
        "et": ("ethyl", "[*:1]CC"),
        "propyl": ("n-propyl", "[*:1]CCC"),
        "n-propyl": ("n-propyl", "[*:1]CCC"),
        "isopropyl": ("isopropyl", "[*:1]C(C)C"),
        "i-propyl": ("isopropyl", "[*:1]C(C)C"),
        "ipr": ("isopropyl", "[*:1]C(C)C"),
        "butyl": ("n-butyl", "[*:1]CCCC"),
        "n-butyl": ("n-butyl", "[*:1]CCCC"),
        "tert-butyl": ("tert-butyl", "[*:1]C(C)(C)C"),
        "t-butyl": ("tert-butyl", "[*:1]C(C)(C)C"),
        "tbu": ("tert-butyl", "[*:1]C(C)(C)C"),
        "phenyl": ("phenyl", "[*:1]c1ccccc1"),
        "ph": ("phenyl", "[*:1]c1ccccc1"),
        "cyano": ("cyano", "[*:1]C#N"),
        "cn": ("cyano", "[*:1]C#N"),
        "nitro": ("nitro", "[*:1][N+](=O)[O-]"),
        "no2": ("nitro", "[*:1][N+](=O)[O-]"),
        "methoxy": ("methoxy", "[*:1]OC"),
        "och3": ("methoxy", "[*:1]OC"),
        "ome": ("methoxy", "[*:1]OC"),
        "ethoxy": ("ethoxy", "[*:1]OCC"),
        "oet": ("ethoxy", "[*:1]OCC"),
        "trifluoromethyl": ("trifluoromethyl", "[*:1]C(F)(F)F"),
        "cf3": ("trifluoromethyl", "[*:1]C(F)(F)F"),
        "carboxyl": ("carboxyl", "[*:1]C(=O)O"),
        "carboxy": ("carboxyl", "[*:1]C(=O)O"),
        "carboxylic acid": ("carboxyl", "[*:1]C(=O)O"),
        "cooh": ("carboxyl", "[*:1]C(=O)O"),
        "co2h": ("carboxyl", "[*:1]C(=O)O"),
        "benzyl": ("benzyl", "[*:1]Cc1ccccc1"),
        "2-pyridyl": ("pyridin-2-yl", "[*:1]c1ccccn1"),
        "3-pyridyl": ("pyridin-3-yl", "[*:1]c1cccnc1"),
        "4-pyridyl": ("pyridin-4-yl", "[*:1]c1ccncc1"),
        "piperidino": ("piperidin-1-yl", "[*:1]N1CCCCC1"),
        "morpholino": ("morpholin-4-yl", "[*:1]N1CCOCC1"),
        "2-fluorobenzyl": ("2-fluorobenzyl", "[*:1]Cc1ccccc1F"),
        "3-fluorobenzyl": ("3-fluorobenzyl", "[*:1]Cc1cccc(F)c1"),
        "4-fluorobenzyl": ("4-fluorobenzyl", "[*:1]Cc1ccc(F)cc1"),
        "2,6-difluorobenzyl": (
            "2,6-difluorobenzyl", "[*:1]Cc1c(F)cccc1F"
        ),
        "2-cyanobenzyl": ("2-cyanobenzyl", "[*:1]Cc1ccccc1C#N"),
        "3-cyanobenzyl": ("3-cyanobenzyl", "[*:1]Cc1cccc(C#N)c1"),
        "sulfonic acid": ("sulfonic acid", "[*:1]S(=O)(=O)O"),
        "sulfonic": ("sulfonic acid", "[*:1]S(=O)(=O)O"),
    }
    if key in fragment_aliases:
        name, smiles = fragment_aliases[key]
        return [_fragment(name, smiles)]

    if key in {"single covalent bond", "single bond", "bond"}:
        return [_candidate(name="single bond", kind="bond")]

    # An unconstrained aryl class is not finite.  Phenyl is emitted only as an
    # explicitly labelled representative, never as a claim of full coverage.
    if key in {"aryl", "aromatic"}:
        return [_fragment("phenyl representative", "[*:1]c1ccccc1", "representative")]

    return []


class FragmentResolver:
    """Resolve textual variable values using curated and bundled sources."""

    def __init__(
        self,
        fragment_library_path: str | os.PathLike | None = (
            DEFAULT_FRAGMENT_LIBRARY_PATH
        ),
        max_candidates_per_value: int | None = None,
    ):
        self.fragment_library_path = (
            str(fragment_library_path) if fragment_library_path else None
        )
        if max_candidates_per_value is None:
            self.max_candidates_per_value = None
        else:
            self.max_candidates_per_value = int(max_candidates_per_value)
            if self.max_candidates_per_value < 1:
                raise ValueError(
                    "max_candidates_per_value must be at least 1 or None"
                )
        self._library_description_index: dict[str, list[dict]] = {}
        self._library_name_index: dict[str, list[dict]] = {}
        self.library_metadata = {
            "enabled": bool(self.fragment_library_path),
            "path": self.fragment_library_path,
            "loaded": False,
            "records": 0,
            "format": "description_name_smiles_json",
            "bundled_with_project": bool(
                self.fragment_library_path
                and Path(self.fragment_library_path).resolve()
                == DEFAULT_FRAGMENT_LIBRARY_PATH.resolve()
            ),
        }
        if self.library_metadata["bundled_with_project"]:
            self.library_metadata["project_relative_path"] = (
                "resources/markush_fragment_library.json"
            )
        self._load_fragment_library()

    def _load_fragment_library(self) -> None:
        if not self.fragment_library_path:
            return
        path = Path(self.fragment_library_path)
        if not path.is_file():
            self.library_metadata["error"] = "library_file_not_found"
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self.library_metadata["error"] = f"{type(exc).__name__}: {exc}"
            return
        if not isinstance(records, list):
            self.library_metadata["error"] = "library_root_is_not_a_list"
            return

        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.library_metadata.update({
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            })
        except OSError as exc:
            self.library_metadata["fingerprint_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        for record in records:
            if not isinstance(record, dict):
                continue
            description = normalize_descriptor(record.get("Description", ""))
            name = normalize_descriptor(record.get("Name", ""))
            if description:
                self._library_description_index.setdefault(description, []).append(record)
            if name:
                self._library_name_index.setdefault(name, []).append(record)
        self.library_metadata.update({"loaded": True, "records": len(records)})

    @staticmethod
    def _canonicalize_fragment(smiles: str) -> str | None:
        # Invalid library records are expected to be skipped and recorded,
        # not printed as thousands of RDKit parser warnings during a batch.
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            return None
        dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy_atoms) != 1 or dummy_atoms[0].GetDegree() != 1:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    def _library_candidates(self, key: str, source_value: str) -> list[FragmentCandidate]:
        records = list(self._library_description_index.get(key, []))
        records.extend(self._library_name_index.get(key, []))
        candidates = []
        for record in records:
            raw_smiles = str(record.get("SMILE", "")).strip()
            if raw_smiles.count("[R]") != 1:
                continue
            fragment_smiles = raw_smiles.replace("[R]", "[*:1]", 1)
            canonical = self._canonicalize_fragment(fragment_smiles)
            if canonical is None:
                continue
            candidates.append(FragmentCandidate(
                source_value=source_value,
                name=str(record.get("Name") or source_value).strip(),
                kind="fragment",
                source="fragment_library",
                coverage="library_mapping",
                fragment_smiles=fragment_smiles,
                canonical_fragment_smiles=canonical,
                library_description=str(record.get("Description") or "").strip(),
            ))
        return candidates

    @staticmethod
    def _range_fallback(key: str, source_value: str) -> list[FragmentCandidate]:
        # Only accept descriptors whose *main class* is the ranged alkyl or
        # alkoxy term.  A substring search would incorrectly interpret a term
        # such as "C1-C4 alkoxy aryl" as a plain alkoxy group.
        match = re.fullmatch(
            r"(?:(?:linear or branched|straight(?:-chain)?|lower|"
            r"substituted or unsubstituted|unsubstituted or substituted)\s+)*"
            r"\(?c(\d+)-c(\d+)\)?\s*(alkyl|alkoxy)"
            r"(?:\s+group)?"
            r"(?:\s+(?:optionally |unsubstituted or )?substituted\s+with\s+.+)?",
            key,
        )
        if not match:
            return []
        start, end = int(match.group(1)), int(match.group(2))
        family = match.group(3)
        if start < 1 or end < start or end > 20:
            return []
        candidates = []
        for carbon_count in range(start, end + 1):
            chain = "C" * carbon_count
            smiles = f"[*:1]{'O' if family == 'alkoxy' else ''}{chain}"
            candidates.append(FragmentCandidate(
                source_value=source_value,
                name=f"straight-chain C{carbon_count} {family}",
                kind="fragment",
                source="range_rule",
                coverage="representative_straight_chain_only",
                fragment_smiles=smiles,
                canonical_fragment_smiles=smiles,
            ))
        return candidates

    def resolve_value(self, value: str) -> tuple[list[FragmentCandidate], list[str]]:
        """Resolve one textual value and return candidates plus audit warnings."""
        source_value = str(value).strip()
        key = normalize_descriptor(source_value)
        warnings = []
        if not key or re.fullmatch(r"\d+(?:-\d+)?", key):
            return [], warnings

        candidates = [
            replace(candidate, source_value=source_value)
            for candidate in _builtin_candidates(key)
        ]
        if not candidates:
            candidates = self._library_candidates(key, source_value)
        if not candidates:
            candidates = self._range_fallback(key, source_value)

        deduplicated = {}
        for candidate in candidates:
            if candidate.kind == "fragment" and candidate.fragment_smiles:
                canonical = self._canonicalize_fragment(
                    candidate.fragment_smiles
                )
                if canonical is None:
                    warnings.append(f"invalid_fragment_skipped:{candidate.name}")
                    continue
                candidate = replace(
                    candidate, canonical_fragment_smiles=canonical
                )
            deduplicated.setdefault(candidate.key(), candidate)
        candidates = sorted(
            deduplicated.values(),
            key=lambda item: (
                item.canonical_fragment_smiles or "",
                item.atom_smiles or "",
                item.name.lower(),
            ),
        )
        if (
            self.max_candidates_per_value is not None
            and len(candidates) > self.max_candidates_per_value
        ):
            warnings.append(
                f"candidate_limit_applied:{len(candidates)}->"
                f"{self.max_candidates_per_value}"
            )
            candidates = candidates[: self.max_candidates_per_value]
        return candidates, warnings

    def resolve_variable(self, values) -> tuple[list[FragmentCandidate], dict]:
        """Resolve all values for one variable and build an auditable report."""
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        input_values = [
            str(value).strip()
            for value in values
            if value is not None and str(value).strip()
        ]
        all_candidates = []
        unresolved = []
        warnings = []
        for value in input_values:
            candidates, value_warnings = self.resolve_value(value)
            warnings.extend(f"{value}:{warning}" for warning in value_warnings)
            if candidates:
                all_candidates.extend(candidates)
            else:
                unresolved.append(value)

        deduplicated = {}
        for candidate in all_candidates:
            deduplicated.setdefault(candidate.key(), candidate)
        candidates = list(deduplicated.values())
        report = {
            "input_values": input_values,
            "candidate_count": len(candidates),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "unresolved_values": unresolved,
            "warnings": warnings,
        }
        return candidates, report
