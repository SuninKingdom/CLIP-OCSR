"""Instantiate a parsed Markush representation as concrete molecular SMILES.

This graph-based implementation replaces the earlier string-substitution
combiner. It supports:

* terminal substituent and internal atom replacement;
* one assignment per variable label, including repeated occurrences;
* finite category expansion through :mod:`fragment_resolver`;
* linear frequency tokens such as ``[(CH2)n]``;
* sulfoxide/sulfone-style ``S[(O)m]`` frequency tokens;
* inferred ring-site enumeration for position variables such as ``[R1$]``;
* RDKit sanitization, canonical isomeric SMILES, global deduplication, and
  explicit enumeration limits.

Unsupported or open-ended semantics are retained in the audit report instead
of being silently treated as hydrogen or as a complete chemical space.
"""

from __future__ import annotations

import itertools
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, rdBase

from clip_ocsr.utils.abbrev_group import abbrevgroup2smiles
try:  # Package import, e.g. ``markush_parsing.markush_instantiator``.
    from .fragment_resolver import FragmentCandidate, FragmentResolver
except ImportError:  # Script execution, e.g. ``python markush_parsing/run.py``.
    from fragment_resolver import FragmentCandidate, FragmentResolver


REPEAT_PATTERN = re.compile(
    r"\[(?P<prefix>[A-Za-z0-9@+\-]*?)"
    r"\((?P<unit>[^()\[\]]+)\)"
    r"(?P<label>[A-Za-z][A-Za-z0-9']*)\]"
)

LIKELY_MARKUSH_LABEL = re.compile(
    r"(?:R[A-Za-z0-9'\"-]*|[XYZABGLMQTUW][A-Za-z0-9'\"-]*|Ar|Cy|Hal)\$?"
)


@dataclass(frozen=True)
class InstantiationLimits:
    max_products: int = 256
    max_assignment_attempts: int = 10000
    max_position_variants: int = 64
    max_frequency_variants: int = 64
    max_repeat_count: int = 100

    def normalized(self) -> "InstantiationLimits":
        return InstantiationLimits(
            max_products=max(1, int(self.max_products)),
            max_assignment_attempts=max(1, int(self.max_assignment_attempts)),
            max_position_variants=max(1, int(self.max_position_variants)),
            max_frequency_variants=max(1, int(self.max_frequency_variants)),
            max_repeat_count=max(0, int(self.max_repeat_count)),
        )


class MarkushInstantiator:
    """Convert a backbone pseudo-SMILES plus variable table to products."""

    schema_version = "1.0"

    def __init__(
        self,
        resolver: FragmentResolver | None = None,
        limits: InstantiationLimits | None = None,
        abbreviation_library_path: str | None = None,
    ):
        self.resolver = resolver or FragmentResolver()
        self.limits = (limits or InstantiationLimits()).normalized()
        self.abbreviation_library_path = abbreviation_library_path or str(
            Path(__file__).resolve().parent.parent
            / "assets"
            / "abbrev_group.json"
        )

    @staticmethod
    def _normalize_variables(variables) -> dict[str, list[str]]:
        if not isinstance(variables, dict):
            return {}
        normalized = {}
        for raw_label, raw_values in variables.items():
            label = str(raw_label).strip()
            if not label:
                continue
            values = (
                raw_values
                if isinstance(raw_values, (list, tuple, set))
                else [raw_values]
            )
            normalized[label] = [
                str(value).strip()
                for value in values
                if value is not None and str(value).strip()
            ]
        return normalized

    @staticmethod
    def _label_key(label: str) -> str:
        """Normalize OCR/LLM label typography for conservative matching.

        This aligns forms such as ``R_c``/``R^c``/``R_{c}`` with the OCSR
        placeholder ``[Rc]``.  It is used only when there is one unambiguous
        label in the backbone; the emitted products retain the backbone label.
        """
        text = unicodedata.normalize("NFKC", str(label or "")).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if text.endswith("$"):
            text = text[:-1]
        text = re.sub(r"[_^]\{([^{}]+)\}", r"\1", text)
        text = re.sub(r"[\s_^{}\\]", "", text)
        # Preserve case: ``N`` may be an atom/linker variable while ``n`` is
        # commonly a repeat count.  Treating them as equal can silently turn
        # a substituent definition into a frequency definition.
        return text

    @classmethod
    def _core_declared_labels(
        cls, pseudo_smiles: str, variable_labels=()
    ) -> list[str]:
        labels = []
        variable_keys = {
            cls._label_key(label) for label in variable_labels
        }
        for match in REPEAT_PATTERN.finditer(pseudo_smiles):
            label = match.group("label")
            if label not in labels:
                labels.append(label)
        for content in re.findall(r"\[([^\[\]]+)\]", pseudo_smiles):
            if content.startswith("("):
                continue
            label = content[:-1] if content.endswith("$") else content
            if not (
                content.endswith("$")
                or LIKELY_MARKUSH_LABEL.fullmatch(content)
                or cls._label_key(label) in variable_keys
            ):
                continue
            if label not in labels:
                labels.append(label)
        return labels

    def _align_variables_to_core(self, pseudo_smiles: str, variables):
        """Align harmless label typography differences without guessing."""
        raw_variables = self._normalize_variables(variables)
        core_labels = self._core_declared_labels(
            pseudo_smiles, raw_variables.keys()
        )
        exact_core_labels = set(core_labels)
        core_by_key = {}
        for label in core_labels:
            core_by_key.setdefault(self._label_key(label), []).append(label)

        aligned = {}
        mapping = []
        warnings = []
        source_labels_by_target = {}
        for raw_label, values in raw_variables.items():
            if raw_label in exact_core_labels:
                target = raw_label
                reason = "exact"
            else:
                matches = core_by_key.get(self._label_key(raw_label), [])
                if len(matches) == 1:
                    target = matches[0]
                    reason = "normalized_typography"
                else:
                    target = raw_label
                    reason = "not_aligned" if not matches else "ambiguous"
                    if len(matches) > 1:
                        warnings.append(
                            f"ambiguous_variable_label:{raw_label}:"
                            + ",".join(matches)
                        )

            if target in source_labels_by_target and (
                raw_label not in source_labels_by_target[target]
            ):
                warnings.append(
                    f"variable_label_collision:{target}:"
                    + ",".join(source_labels_by_target[target] + [raw_label])
                )
            source_labels_by_target.setdefault(target, []).append(raw_label)
            aligned.setdefault(target, [])
            for value in values:
                if value not in aligned[target]:
                    aligned[target].append(value)
            mapping.append({
                "input_label": raw_label,
                "backbone_label": target,
                "method": reason,
            })

        return aligned, {
            "core_labels_seen": core_labels,
            "mapping": mapping,
            "warnings": warnings,
        }

    def _expand_composite_placeholders(self, pseudo_smiles: str, variables):
        """Expand common OCSR tokens such as ``[OR2]`` to ``O[R2]``.

        The rewrite is applied only when the suffix maps unambiguously to an
        extracted variable label and the complete bracket token is not itself
        an extracted label.  This avoids treating ordinary abbreviations such
        as ``[OCH3]`` as composite variables.
        """
        raw_variables = self._normalize_variables(variables)
        exact_labels = set(raw_variables)
        labels_by_key = {}
        for label in raw_variables:
            labels_by_key.setdefault(self._label_key(label), []).append(label)

        expansions = []
        warnings = []
        prefixes = (
            ("CH2", "[CH2]"),
            ("NH", "[NH]"),
            ("O", "O"),
            ("S", "S"),
            ("N", "N"),
        )

        def replacement(match: re.Match) -> str:
            content = match.group(1)
            position_marker = "$" if content.endswith("$") else ""
            label = content[:-1] if position_marker else content
            if label in exact_labels:
                return match.group(0)
            for prefix, rendered_prefix in prefixes:
                if not label.startswith(prefix) or len(label) == len(prefix):
                    continue
                suffix = label[len(prefix):]
                matches = labels_by_key.get(self._label_key(suffix), [])
                if len(matches) != 1:
                    if len(matches) > 1:
                        warnings.append(
                            f"ambiguous_composite_placeholder:{content}"
                        )
                    continue
                expanded = f"{rendered_prefix}[{suffix}{position_marker}]"
                expansions.append({
                    "input_token": match.group(0),
                    "expanded_token": expanded,
                    "prefix": prefix,
                    "variable_label": suffix,
                })
                return expanded
            return match.group(0)

        expanded_smiles = re.sub(
            r"\[([^\[\]]+)\]", replacement, pseudo_smiles
        )
        return expanded_smiles, {
            "composite_placeholder_expansions": expansions,
            "warnings": warnings,
        }

    @staticmethod
    def _label_pattern(label: str) -> re.Pattern:
        return re.compile(r"\[" + re.escape(label) + r"(?P<position>\$)?\]")

    def _structural_labels(self, smiles_variants: list[str], variables: dict) -> list[str]:
        first_positions = {}
        for label in variables:
            pattern = self._label_pattern(label)
            positions = [
                match.start()
                for smiles in smiles_variants
                for match in pattern.finditer(smiles)
            ]
            if positions:
                first_positions[label] = min(positions)
        return sorted(first_positions, key=lambda label: (first_positions[label], label))

    @staticmethod
    def _integer_values(values) -> tuple[list[int], list[str]]:
        integers = []
        invalid = []
        for value in values:
            text = str(value).strip()
            if re.fullmatch(r"\d+", text):
                integers.append(int(text))
                continue
            range_match = re.fullmatch(
                r"(\d+)\s*(?:-|to)\s*(\d+)", text, re.IGNORECASE
            )
            if range_match:
                start, end = (int(item) for item in range_match.groups())
                integers.extend(range(min(start, end), max(start, end) + 1))
            else:
                invalid.append(text)
        return sorted(set(integers)), invalid

    @staticmethod
    def _is_sulfur_oxidation_context(prefix: str) -> bool:
        return bool(re.search(
            r"(?:S|\[S[^\]]*\])(?:\([^()]*\))*$",
            prefix,
        ))

    def _render_repeat(self, source: str, match: re.Match, count: int) -> str | None:
        prefix = match.group("prefix")
        unit = re.sub(r"\s+", "", match.group("unit"))
        if unit == "CH2":
            repeated = "C" * count
        elif unit in {"C", "N", "O", "S"}:
            if unit == "O" and self._is_sulfur_oxidation_context(
                source[: match.start()]
            ):
                return prefix + "(=O)" * count
            repeated = unit * count
        else:
            return None
        return prefix + repeated

    def _expand_frequency(self, pseudo_smiles: str, variables: dict) -> tuple[list[dict], dict]:
        matches = list(REPEAT_PATTERN.finditer(pseudo_smiles))
        report = {
            "tokens": [],
            "frequency_labels": [],
            "theoretical_assignments": 1,
            "generated_variants": 1,
            "truncated": False,
            "unsupported_tokens": [],
            "invalid_values": {},
            "values_above_repeat_limit": {},
        }
        if not matches:
            return [{"smiles": pseudo_smiles, "assignment": {}}], report

        token_labels = []
        unsupported_units = set()
        for match in matches:
            token = match.group(0)
            label = match.group("label")
            unit = match.group("unit")
            report["tokens"].append({
                "token": token,
                "prefix": match.group("prefix"),
                "unit": unit,
                "count_label": label,
            })
            if label not in token_labels:
                token_labels.append(label)
            if self._render_repeat(pseudo_smiles, match, 1) is None:
                unsupported_units.add(token)

        report["frequency_labels"] = token_labels
        report["unsupported_tokens"] = sorted(unsupported_units)
        if unsupported_units:
            return [], report

        values_by_label = {}
        for label in token_labels:
            integers, invalid = self._integer_values(variables.get(label, []))
            if invalid:
                report["invalid_values"][label] = invalid
            above_limit = [
                value for value in integers if value > self.limits.max_repeat_count
            ]
            allowed = [
                value for value in integers if value <= self.limits.max_repeat_count
            ]
            if above_limit:
                report["values_above_repeat_limit"][label] = above_limit
            if not allowed:
                report["unsupported_tokens"].append(
                    f"{label}:no_supported_integer_counts"
                )
                return [], report
            values_by_label[label] = allowed

        report["theoretical_assignments"] = math.prod(
            len(values_by_label[label]) for label in token_labels
        )
        variants = []
        combinations = itertools.product(
            *(values_by_label[label] for label in token_labels)
        )
        for counts in itertools.islice(
            combinations, self.limits.max_frequency_variants
        ):
            assignment = dict(zip(token_labels, counts))

            def replacement(match: re.Match) -> str:
                return self._render_repeat(
                    pseudo_smiles, match, assignment[match.group("label")]
                )

            variants.append({
                "smiles": REPEAT_PATTERN.sub(replacement, pseudo_smiles),
                "assignment": assignment,
            })

        report["generated_variants"] = len(variants)
        report["truncated"] = (
            report["theoretical_assignments"] > len(variants)
        )
        return variants, report

    @staticmethod
    def _unresolved_core_labels(smiles: str, known_labels: set[str]) -> list[str]:
        unresolved = set()
        for content in re.findall(r"\[([^\[\]]+)\]", smiles):
            if content.startswith("("):
                continue
            # ``test_markush`` currently uses Python 3.8, where
            # ``str.removesuffix`` is unavailable.
            label = content[:-1] if content.endswith("$") else content
            if label in known_labels:
                continue
            if LIKELY_MARKUSH_LABEL.fullmatch(content):
                unresolved.add(content)
        return sorted(unresolved)

    def _encode_core(self, pseudo_smiles: str, structural_labels: list[str]):
        encoded = pseudo_smiles
        occurrence_map = {}
        next_map = 1001
        for label in structural_labels:
            pattern = self._label_pattern(label)

            def replacement(match: re.Match) -> str:
                nonlocal next_map
                atom_map = next_map
                next_map += 1
                occurrence_map[atom_map] = {
                    "label": label,
                    "position_variable": bool(match.group("position")),
                }
                return f"[*:{atom_map}]"

            encoded = pattern.sub(replacement, encoded)

        encoded = abbrevgroup2smiles(
            encoded, self.abbreviation_library_path
        )
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(encoded)
        if mol is None:
            return None, occurrence_map, encoded
        for atom in mol.GetAtoms():
            metadata = occurrence_map.get(atom.GetAtomMapNum())
            if metadata is None:
                continue
            atom.SetProp("_markush_label", metadata["label"])
            atom.SetBoolProp(
                "_markush_position_variable", metadata["position_variable"]
            )
            if metadata["position_variable"] and atom.GetDegree() == 1:
                atom.SetIntProp(
                    "_markush_position_site",
                    atom.GetNeighbors()[0].GetIdx(),
                )
        return mol, occurrence_map, encoded

    @staticmethod
    def _core_key(mol: Chem.Mol) -> str:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    @staticmethod
    def _ring_system_sites(mol: Chem.Mol, anchor_index: int) -> list[int]:
        atom_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
        selected = [ring for ring in atom_rings if anchor_index in ring]
        if not selected:
            return [anchor_index]
        system = set().union(*selected)
        changed = True
        while changed:
            changed = False
            for ring in atom_rings:
                if ring & system and not ring <= system:
                    system.update(ring)
                    changed = True
        return sorted(system)

    def _expand_one_position_dummy(self, mol: Chem.Mol, atom_map: int) -> tuple[list[Chem.Mol], dict]:
        dummy = next(
            (
                atom for atom in mol.GetAtoms()
                if atom.GetAtomMapNum() == atom_map
                and atom.HasProp("_markush_position_variable")
            ),
            None,
        )
        if dummy is None or dummy.GetDegree() != 1:
            return [mol], {
                "atom_map": atom_map,
                "mode": "original_site_only",
                "reason": "position_dummy_not_terminal",
                "candidate_sites": 1,
            }
        anchor = dummy.GetNeighbors()[0]
        anchor_index = anchor.GetIdx()
        bond_type = mol.GetBondBetweenAtoms(
            dummy.GetIdx(), anchor_index
        ).GetBondType()
        sites = self._ring_system_sites(mol, anchor_index)
        candidates = []
        for site in sites:
            if mol.GetAtomWithIdx(site).GetAtomicNum() == 0:
                continue
            rw_mol = Chem.RWMol(mol)
            rw_mol.RemoveBond(dummy.GetIdx(), anchor_index)
            rw_mol.AddBond(dummy.GetIdx(), site, bond_type)
            rw_mol.GetAtomWithIdx(dummy.GetIdx()).SetIntProp(
                "_markush_position_site", site
            )
            candidate = rw_mol.GetMol()
            try:
                with rdBase.BlockLogs():
                    Chem.SanitizeMol(candidate)
            except Exception:
                continue
            candidates.append(candidate)
        if not candidates:
            candidates = [mol]
        deduplicated = {}
        for candidate in candidates:
            deduplicated.setdefault(self._core_key(candidate), candidate)
        candidates = list(deduplicated.values())
        return candidates, {
            "atom_map": atom_map,
            "label": dummy.GetProp("_markush_label")
            if dummy.HasProp("_markush_label") else None,
            "mode": "inferred_ring_system"
            if len(sites) > 1 else "original_site_only",
            "candidate_sites": len(sites),
            "valid_unique_sites": len(candidates),
        }

    def _expand_positions(self, mol: Chem.Mol) -> tuple[list[Chem.Mol], list[dict], bool]:
        position_maps = sorted(
            atom.GetAtomMapNum()
            for atom in mol.GetAtoms()
            if atom.HasProp("_markush_position_variable")
            and atom.GetBoolProp("_markush_position_variable")
        )
        variants = [mol]
        reports = []
        truncated = False
        for atom_map in position_maps:
            expanded = []
            local_reports = []
            for variant in variants:
                candidates, report = self._expand_one_position_dummy(
                    variant, atom_map
                )
                expanded.extend(candidates)
                local_reports.append(report)
            deduplicated = {}
            for candidate in expanded:
                deduplicated.setdefault(self._core_key(candidate), candidate)
            expanded = list(deduplicated.values())
            if len(expanded) > self.limits.max_position_variants:
                expanded = expanded[: self.limits.max_position_variants]
                truncated = True
            variants = expanded
            reports.append({
                "atom_map": atom_map,
                "input_variants": len(local_reports),
                "output_variants": len(variants),
                "details": local_reports,
            })
        return variants, reports, truncated

    @staticmethod
    def _matching_dummies(mol: Chem.Mol, label: str) -> list[Chem.Atom]:
        return [
            atom for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 0
            and atom.HasProp("_markush_label")
            and atom.GetProp("_markush_label") == label
        ]

    @staticmethod
    def _position_assignment(mol: Chem.Mol) -> list[dict]:
        assignment = []
        for atom in mol.GetAtoms():
            if not (
                atom.GetAtomicNum() == 0
                and atom.HasProp("_markush_position_variable")
                and atom.GetBoolProp("_markush_position_variable")
            ):
                continue
            site = (
                atom.GetIntProp("_markush_position_site")
                if atom.HasProp("_markush_position_site")
                else None
            )
            assignment.append({
                "label": atom.GetProp("_markush_label")
                if atom.HasProp("_markush_label") else None,
                "placeholder_atom_map": atom.GetAtomMapNum(),
                "backbone_atom_index": site,
                "backbone_atom_symbol": (
                    mol.GetAtomWithIdx(site).GetSymbol()
                    if site is not None and site < mol.GetNumAtoms()
                    else None
                ),
            })
        return sorted(
            assignment,
            key=lambda item: item["placeholder_atom_map"],
        )

    @staticmethod
    def _clear_markush_properties(atom: Chem.Atom) -> None:
        atom.SetAtomMapNum(0)
        for prop in (
            "_markush_label",
            "_markush_position_variable",
            "_markush_position_site",
        ):
            if atom.HasProp(prop):
                atom.ClearProp(prop)

    @staticmethod
    def _sanitize(mol: Chem.Mol) -> Chem.Mol | None:
        try:
            mol.UpdatePropertyCache(strict=False)
            with rdBase.BlockLogs():
                Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            return mol
        except Exception:
            return None

    def _apply_atom(self, mol: Chem.Mol, label: str, atom_smiles: str) -> Chem.Mol | None:
        with rdBase.BlockLogs():
            template = Chem.MolFromSmiles(atom_smiles)
        if template is None or template.GetNumAtoms() != 1:
            return None
        template_atom = template.GetAtomWithIdx(0)
        rw_mol = Chem.RWMol(mol)
        matched = self._matching_dummies(rw_mol, label)
        if not matched:
            return None
        for atom in matched:
            atom.SetAtomicNum(template_atom.GetAtomicNum())
            atom.SetFormalCharge(template_atom.GetFormalCharge())
            atom.SetIsotope(template_atom.GetIsotope())
            atom.SetNoImplicit(template_atom.GetNoImplicit())
            self._clear_markush_properties(atom)
        return self._sanitize(rw_mol.GetMol())

    def _apply_hydrogen(self, mol: Chem.Mol, label: str) -> Chem.Mol | None:
        rw_mol = Chem.RWMol(mol)
        matched = self._matching_dummies(rw_mol, label)
        if not matched or any(atom.GetDegree() != 1 for atom in matched):
            return None
        # Materialize H first, then let RDKit remove it while updating the
        # neighbor's implicit/explicit-H and tetrahedral-stereo state.  Simply
        # deleting a dummy from bracketed ``[C@]`` can leave a carbon radical
        # and erase a valid stereocenter.
        for atom in matched:
            atom.SetAtomicNum(1)
            atom.SetFormalCharge(0)
            atom.SetIsotope(0)
            atom.SetNoImplicit(True)
            self._clear_markush_properties(atom)
        materialized = self._sanitize(rw_mol.GetMol())
        if materialized is None:
            return None
        try:
            with rdBase.BlockLogs():
                collapsed = Chem.RemoveHs(materialized)
        except Exception:
            return None
        return self._sanitize(collapsed)

    def _apply_bond(self, mol: Chem.Mol, label: str) -> Chem.Mol | None:
        current = Chem.Mol(mol)
        while True:
            matched = self._matching_dummies(current, label)
            if not matched:
                break
            dummy = matched[0]
            if dummy.GetDegree() != 2:
                return None
            neighbors = [atom.GetIdx() for atom in dummy.GetNeighbors()]
            rw_mol = Chem.RWMol(current)
            if rw_mol.GetBondBetweenAtoms(*neighbors) is None:
                rw_mol.AddBond(neighbors[0], neighbors[1], Chem.BondType.SINGLE)
            rw_mol.RemoveAtom(dummy.GetIdx())
            current = self._sanitize(rw_mol.GetMol())
            if current is None:
                return None
        return current

    @staticmethod
    def _attachment_bond_type(core_bond, fragment_bond):
        single = Chem.BondType.SINGLE
        core_type = core_bond.GetBondType()
        fragment_type = fragment_bond.GetBondType()
        if core_type != single and fragment_type != single and core_type != fragment_type:
            return None
        return core_type if core_type != single else fragment_type

    def _apply_fragment(self, mol: Chem.Mol, label: str, smiles: str) -> Chem.Mol | None:
        with rdBase.BlockLogs():
            fragment = Chem.MolFromSmiles(smiles)
        if fragment is None:
            return None
        fragment_dummies = [
            atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0
        ]
        if len(fragment_dummies) != 1 or fragment_dummies[0].GetDegree() != 1:
            return None
        fragment_dummy = fragment_dummies[0]
        fragment_neighbor = fragment_dummy.GetNeighbors()[0]
        fragment_bond = fragment.GetBondBetweenAtoms(
            fragment_dummy.GetIdx(), fragment_neighbor.GetIdx()
        )

        current = Chem.Mol(mol)
        while True:
            matched = self._matching_dummies(current, label)
            if not matched:
                break
            core_dummy = matched[0]
            if core_dummy.GetDegree() != 1:
                return None
            core_neighbor = core_dummy.GetNeighbors()[0]
            core_bond = current.GetBondBetweenAtoms(
                core_dummy.GetIdx(), core_neighbor.GetIdx()
            )
            bond_type = self._attachment_bond_type(core_bond, fragment_bond)
            if bond_type is None:
                return None

            # Graft the fragment *onto* the existing core dummy atom instead
            # of deleting that atom and appending a replacement.  Retaining
            # the core atom index preserves the neighbor ordering used by any
            # adjacent tetrahedral stereocenter.  This matters when two or
            # more fragment-valued variables meet at the same chiral atom.
            rw_mol = Chem.RWMol(current)
            replacement_atom = Chem.Atom(fragment_neighbor)
            replacement_atom.SetAtomMapNum(0)
            rw_mol.ReplaceAtom(
                core_dummy.GetIdx(), replacement_atom, preserveProps=False
            )
            self._clear_markush_properties(
                rw_mol.GetAtomWithIdx(core_dummy.GetIdx())
            )
            retained_core_bond = rw_mol.GetBondBetweenAtoms(
                core_dummy.GetIdx(), core_neighbor.GetIdx()
            )
            retained_core_bond.SetBondType(bond_type)
            if fragment_bond.GetBondDir() != Chem.BondDir.NONE:
                retained_core_bond.SetBondDir(fragment_bond.GetBondDir())

            fragment_to_product = {
                fragment_neighbor.GetIdx(): core_dummy.GetIdx(),
                # The fragment dummy is not copied, but for alkene stereo it
                # corresponds exactly to the core-side attachment atom.
                fragment_dummy.GetIdx(): core_neighbor.GetIdx(),
            }
            for atom in fragment.GetAtoms():
                if atom.GetIdx() in {
                    fragment_dummy.GetIdx(),
                    fragment_neighbor.GetIdx(),
                }:
                    continue
                copied_atom = Chem.Atom(atom)
                copied_atom.SetAtomMapNum(0)
                fragment_to_product[atom.GetIdx()] = rw_mol.AddAtom(
                    copied_atom
                )

            copied_bonds = []
            for bond in fragment.GetBonds():
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                if fragment_dummy.GetIdx() in {begin, end}:
                    continue
                product_begin = fragment_to_product[begin]
                product_end = fragment_to_product[end]
                rw_mol.AddBond(
                    product_begin, product_end, bond.GetBondType()
                )
                product_bond = rw_mol.GetBondBetweenAtoms(
                    product_begin, product_end
                )
                product_bond.SetIsAromatic(bond.GetIsAromatic())
                product_bond.SetIsConjugated(bond.GetIsConjugated())
                product_bond.SetBondDir(bond.GetBondDir())
                copied_bonds.append((bond, product_bond))

            # Bond stereo references atom indices, so copy it only after all
            # fragment atoms have been mapped into the product graph.
            for source_bond, product_bond in copied_bonds:
                stereo_atoms = list(source_bond.GetStereoAtoms())
                if stereo_atoms and all(
                    atom_index in fragment_to_product
                    for atom_index in stereo_atoms
                ):
                    product_bond.SetStereoAtoms(
                        fragment_to_product[stereo_atoms[0]],
                        fragment_to_product[stereo_atoms[1]],
                    )
                    product_bond.SetStereo(source_bond.GetStereo())
            current = self._sanitize(rw_mol.GetMol())
            if current is None:
                return None
        return current

    def _apply_candidate(
        self, mol: Chem.Mol, label: str, candidate: FragmentCandidate
    ) -> Chem.Mol | None:
        if candidate.kind == "atom" and candidate.atom_smiles:
            return self._apply_atom(mol, label, candidate.atom_smiles)
        if candidate.kind == "hydrogen":
            return self._apply_hydrogen(mol, label)
        if candidate.kind == "bond":
            return self._apply_bond(mol, label)
        if candidate.kind == "fragment" and candidate.fragment_smiles:
            return self._apply_fragment(mol, label, candidate.fragment_smiles)
        return None

    @staticmethod
    def _final_smiles(mol: Chem.Mol) -> str | None:
        if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            return None
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            return Chem.MolToSmiles(
                mol, canonical=True, isomericSmiles=True
            )
        except Exception:
            return None

    @staticmethod
    def _candidate_assignment(candidate: FragmentCandidate) -> dict:
        return candidate.to_dict()

    def instantiate(self, pseudo_smiles: str, variables) -> dict:
        """Instantiate one parsed Markush representation.

        The returned object is deliberately verbose so every approximation,
        skipped value, invalid combination, and enumeration limit remains
        auditable for manuscript experiments.
        """
        result = {
            "schema_version": self.schema_version,
            "status": "failed",
            "failure_reason": None,
            "is_fully_enumerated": False,
            "input_pseudo_smiles": pseudo_smiles,
            "normalized_pseudo_smiles": None,
            "input_variables": variables,
            "limits": {
                "max_products": self.limits.max_products,
                "max_assignment_attempts": self.limits.max_assignment_attempts,
                "max_position_variants": self.limits.max_position_variants,
                "max_frequency_variants": self.limits.max_frequency_variants,
                "max_repeat_count": self.limits.max_repeat_count,
            },
            "fragment_library": dict(self.resolver.library_metadata),
            "core_normalization": {},
            "variable_label_alignment": {},
            "resolved_variables": {},
            "nonstructural_variables": {},
            "frequency": {},
            "position_variation": {
                "present": False,
                "strategy": "none",
                "reports": [],
                "truncated": False,
            },
            "unresolved_core_labels": [],
            "unresolved_values": {},
            "backbone_variants": 0,
            "backbone_parse_failures": 0,
            "theoretical_product_combinations": 0,
            "attempted_combinations": 0,
            "invalid_combinations": 0,
            "duplicate_products_removed": 0,
            "product_count": 0,
            "truncated": False,
            "products": [],
            "errors": [],
            "warnings": [],
        }
        if not isinstance(pseudo_smiles, str) or not pseudo_smiles.strip():
            result["failure_reason"] = "missing_pseudo_smiles"
            result["warnings"].append("missing_pseudo_smiles")
            return result
        pseudo_smiles = pseudo_smiles.strip()
        if pseudo_smiles.upper() == "EMPTY":
            result["failure_reason"] = "empty_structure_sentinel"
            result["warnings"].append("empty_structure_sentinel")
            return result

        pseudo_smiles, core_normalization = (
            self._expand_composite_placeholders(pseudo_smiles, variables)
        )
        result["normalized_pseudo_smiles"] = pseudo_smiles
        result["core_normalization"] = core_normalization
        result["warnings"].extend(core_normalization["warnings"])

        normalized_variables, alignment_report = self._align_variables_to_core(
            pseudo_smiles, variables
        )
        result["variable_label_alignment"] = alignment_report
        result["warnings"].extend(alignment_report["warnings"])
        frequency_variants, frequency_report = self._expand_frequency(
            pseudo_smiles, normalized_variables
        )
        result["frequency"] = frequency_report
        if not frequency_variants:
            result["failure_reason"] = "frequency_expansion_failed"
            result["warnings"].append("frequency_expansion_failed")
            return result

        variant_smiles = [item["smiles"] for item in frequency_variants]
        structural_labels = self._structural_labels(
            variant_smiles, normalized_variables
        )
        frequency_labels = set(frequency_report.get("frequency_labels", []))
        result["nonstructural_variables"] = {
            label: values
            for label, values in normalized_variables.items()
            if label not in structural_labels and label not in frequency_labels
        }

        unresolved_core_labels = sorted(set().union(*(
            set(self._unresolved_core_labels(smiles, set(structural_labels)))
            for smiles in variant_smiles
        )))
        result["unresolved_core_labels"] = unresolved_core_labels
        if unresolved_core_labels:
            result["failure_reason"] = "unresolved_labels_remain_in_backbone"
            result["warnings"].append("unresolved_labels_remain_in_backbone")
            return result

        candidates_by_label = {}
        unresolved_values_present = False
        incomplete_coverage_present = False
        for label in structural_labels:
            candidates, report = self.resolver.resolve_variable(
                normalized_variables.get(label, [])
            )
            result["resolved_variables"][label] = report
            candidates_by_label[label] = candidates
            if any(
                "candidate_limit_applied" in warning
                for warning in report.get("warnings", [])
            ):
                result["truncated"] = True
                result["warnings"].append(
                    f"fragment_candidate_limit_applied:{label}"
                )
            if report["unresolved_values"]:
                unresolved_values_present = True
                result["unresolved_values"][label] = list(
                    report["unresolved_values"]
                )
            if any(
                candidate.coverage not in {"exact", "complete_finite_class"}
                for candidate in candidates
            ):
                incomplete_coverage_present = True
            if not candidates:
                result["failure_reason"] = (
                    f"no_concrete_candidates_for_structural_label:{label}"
                )
                result["warnings"].append(
                    f"no_concrete_candidates_for_structural_label:{label}"
                )
                return result

        backbone_records = []
        for frequency_variant in frequency_variants:
            mol, occurrence_map, encoded = self._encode_core(
                frequency_variant["smiles"], structural_labels
            )
            if mol is None:
                result["backbone_parse_failures"] += 1
                result["warnings"].append(
                    f"backbone_parse_failed:{encoded}"
                )
                continue
            position_variants, position_reports, position_truncated = (
                self._expand_positions(mol)
            )
            if position_reports:
                result["position_variation"]["present"] = True
                result["position_variation"]["strategy"] = (
                    "inferred_same_ring_system"
                )
                result["position_variation"]["reports"].extend(position_reports)
            result["position_variation"]["truncated"] |= position_truncated
            for position_index, position_mol in enumerate(position_variants):
                backbone_records.append({
                    "mol": position_mol,
                    "frequency_assignment": frequency_variant["assignment"],
                    "position_variant_index": position_index,
                    "position_assignment": self._position_assignment(
                        position_mol
                    ),
                    "occurrence_count": len(occurrence_map),
                })

        # Deduplicate backbones while retaining the first deterministic audit
        # record for equivalent frequency/position expansions.
        deduplicated_backbones = {}
        for record in backbone_records:
            key = self._core_key(record["mol"])
            deduplicated_backbones.setdefault(key, record)
        backbone_records = list(deduplicated_backbones.values())
        result["backbone_variants"] = len(backbone_records)
        if not backbone_records:
            result["failure_reason"] = "no_parseable_backbone_variants"
            result["warnings"].append("no_parseable_backbone_variants")
            return result

        candidate_counts = [
            len(candidates_by_label[label]) for label in structural_labels
        ]
        assignments_per_backbone = math.prod(candidate_counts) if candidate_counts else 1
        theoretical = len(backbone_records) * assignments_per_backbone
        result["theoretical_product_combinations"] = theoretical

        products_by_smiles = {}
        stop = False
        for backbone_index, backbone in enumerate(backbone_records):
            candidate_products = (
                itertools.product(
                    *(candidates_by_label[label] for label in structural_labels)
                )
                if structural_labels
                else [()]
            )
            for selected in candidate_products:
                if result["attempted_combinations"] >= self.limits.max_assignment_attempts:
                    result["truncated"] = True
                    stop = True
                    break
                result["attempted_combinations"] += 1
                product_mol = Chem.Mol(backbone["mol"])
                assignment = {}
                valid = True
                for label, candidate in zip(structural_labels, selected):
                    product_mol = self._apply_candidate(
                        product_mol, label, candidate
                    )
                    assignment[label] = self._candidate_assignment(candidate)
                    if product_mol is None:
                        valid = False
                        break
                if not valid:
                    result["invalid_combinations"] += 1
                    continue
                smiles = self._final_smiles(product_mol)
                if smiles is None:
                    result["invalid_combinations"] += 1
                    continue
                product_record = {
                    "smiles": smiles,
                    "backbone_variant_index": backbone_index,
                    "position_variant_index": backbone["position_variant_index"],
                    "position_assignment": backbone["position_assignment"],
                    "frequency_assignment": backbone["frequency_assignment"],
                    "variable_assignment": assignment,
                }
                if smiles in products_by_smiles:
                    result["duplicate_products_removed"] += 1
                    continue
                products_by_smiles[smiles] = product_record
                if len(products_by_smiles) >= self.limits.max_products:
                    result["truncated"] |= (
                        result["attempted_combinations"] < theoretical
                    )
                    stop = True
                    break
            if stop:
                break

        result["products"] = list(products_by_smiles.values())
        result["product_count"] = len(result["products"])
        result["truncated"] |= bool(
            frequency_report.get("truncated")
            or result["position_variation"]["truncated"]
        )
        if not result["products"]:
            result["failure_reason"] = "no_valid_concrete_products"
            result["warnings"].append("no_valid_concrete_products")
            return result

        partial_reasons = []
        if unresolved_values_present:
            partial_reasons.append("some_variable_values_unresolved")
        if incomplete_coverage_present:
            partial_reasons.append("representative_or_library_mapping_used")
        if alignment_report.get("warnings"):
            partial_reasons.append("variable_label_alignment_not_unambiguous")
        if result["nonstructural_variables"]:
            partial_reasons.append("variable_labels_not_present_in_backbone")
        if result["backbone_parse_failures"]:
            partial_reasons.append("some_backbone_variants_failed_to_parse")
        if result["invalid_combinations"]:
            partial_reasons.append("chemically_invalid_combinations_skipped")
        if result["truncated"]:
            partial_reasons.append("enumeration_truncated")
        if frequency_report.get("invalid_values"):
            partial_reasons.append("invalid_frequency_values_skipped")
        if frequency_report.get("values_above_repeat_limit"):
            partial_reasons.append("frequency_values_above_repeat_limit")
        if result["position_variation"]["present"]:
            partial_reasons.append("position_sites_inferred_from_ring_system")
        result["warnings"].extend(partial_reasons)
        result["status"] = "partial" if partial_reasons else "complete"
        result["failure_reason"] = None
        result["is_fully_enumerated"] = result["status"] == "complete"
        return result


def instantiate_markush(
    pseudo_smiles: str,
    variables,
    *,
    resolver: FragmentResolver | None = None,
    limits: InstantiationLimits | None = None,
) -> dict:
    """Functional convenience wrapper for one Markush representation."""
    return MarkushInstantiator(resolver=resolver, limits=limits).instantiate(
        pseudo_smiles, variables
    )


def build_markush_instantiator(config) -> MarkushInstantiator:
    """Build an instantiator from the public pipeline configuration."""
    resolver = FragmentResolver(
        fragment_library_path=config.fragment_library_path or None,
        max_candidates_per_value=(
            config.instantiation_max_candidates_per_value
        ),
    )
    limits = InstantiationLimits(
        max_products=config.instantiation_max_products,
        max_assignment_attempts=config.instantiation_max_assignment_attempts,
        max_position_variants=config.instantiation_max_position_variants,
        max_frequency_variants=config.instantiation_max_frequency_variants,
        max_repeat_count=config.instantiation_max_repeat_count,
    )
    return MarkushInstantiator(
        resolver=resolver,
        limits=limits,
        abbreviation_library_path=config.abbrev_group_path or None,
    )
