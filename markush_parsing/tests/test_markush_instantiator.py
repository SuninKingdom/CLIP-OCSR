import json
import os
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem

from fragment_resolver import (
    DEFAULT_FRAGMENT_LIBRARY_PATH,
    FragmentResolver,
)
from markush_instantiator import MarkushInstantiator, InstantiationLimits


class FragmentResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolver = FragmentResolver(fragment_library_path=None)

    def test_default_fragment_library_is_bundled_with_project(self):
        self.assertTrue(DEFAULT_FRAGMENT_LIBRARY_PATH.is_file())
        resolver = FragmentResolver()
        self.assertTrue(resolver.library_metadata["bundled_with_project"])
        self.assertTrue(resolver.library_metadata["loaded"])
        self.assertEqual(resolver.library_metadata["records"], 7627)
        self.assertEqual(
            resolver.library_metadata["sha256"],
            "9c0e53d796799aed181a0b854ab5dfae2154ffd016dc56e1a6e93f23b75c417f",
        )
        self.assertEqual(
            resolver.library_metadata["project_relative_path"],
            "resources/markush_fragment_library.json",
        )

    def test_halogen_is_a_complete_finite_class(self):
        candidates, warnings = self.resolver.resolve_value("a halogen atom")
        self.assertEqual(warnings, [])
        self.assertEqual(
            {candidate.atom_smiles for candidate in candidates},
            {"F", "Cl", "Br", "I"},
        )
        self.assertTrue(
            all(
                candidate.coverage == "complete_finite_class"
                for candidate in candidates
            )
        )

    def test_ranged_fallback_does_not_match_a_larger_class_by_substring(self):
        candidates, _ = self.resolver.resolve_value("C1-C4 alkoxy aryl")
        self.assertEqual(candidates, [])

    def test_custom_library_is_optional_validated_and_fingerprinted(self):
        records = [
            {
                "Description": "example finite class",
                "Name": "methyl",
                "SMILE": "[R]C",
            },
            {
                "Description": "example finite class",
                "Name": "ethyl",
                "SMILE": "[R]CC",
            },
            {
                "Description": "example finite class",
                "Name": "invalid two connectors",
                "SMILE": "[R]C[R]",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "fragments.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(records, handle)
            resolver = FragmentResolver(path)
            candidates, _ = resolver.resolve_value("example finite class")

        self.assertEqual(len(candidates), 2)
        self.assertTrue(resolver.library_metadata["loaded"])
        self.assertEqual(len(resolver.library_metadata["sha256"]), 64)
        self.assertTrue(
            all(
                candidate.coverage == "library_mapping"
                for candidate in candidates
            )
        )


class MarkushInstantiatorTests(unittest.TestCase):
    def setUp(self):
        self.instantiator = MarkushInstantiator(
            FragmentResolver(fragment_library_path=None)
        )

    @staticmethod
    def smiles_set(result):
        return {product["smiles"] for product in result["products"]}

    def test_default_abbreviation_library_is_inside_repository(self):
        path = Path(self.instantiator.abbreviation_library_path).resolve()
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "abbrev_group.json")
        self.assertEqual(path.parent.name, "assets")

    def test_halogen_instantiation(self):
        result = self.instantiator.instantiate(
            "c1ccccc1[R1]", {"R1": ["halogen"]}
        )
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["is_fully_enumerated"])
        self.assertEqual(
            self.smiles_set(result),
            {
                "Fc1ccccc1",
                "Clc1ccccc1",
                "Brc1ccccc1",
                "Ic1ccccc1",
            },
        )

    def test_cartesian_product_across_variables(self):
        result = self.instantiator.instantiate(
            "c1cc([R1])ccc1[R2]",
            {"R1": ["F", "Cl"], "R2": ["methyl", "ethyl"]},
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["theoretical_product_combinations"], 4)
        self.assertEqual(result["product_count"], 4)

    def test_repeated_occurrences_share_one_assignment(self):
        result = self.instantiator.instantiate(
            "[R1]CC[R1]", {"R1": ["halogen"]}
        )
        self.assertEqual(
            self.smiles_set(result),
            {"FCCF", "ClCCCl", "BrCCBr", "ICCI"},
        )
        self.assertNotIn("FCCCl", self.smiles_set(result))

    def test_internal_atom_hydrogen_and_deuterium_replacements(self):
        internal = self.instantiator.instantiate(
            "CC[X]CC", {"X": ["oxygen", "sulfur"]}
        )
        hydrogen = self.instantiator.instantiate(
            "c1ccccc1[R1]", {"R1": ["H"]}
        )
        deuterium = self.instantiator.instantiate(
            "c1ccccc1[R1]", {"R1": ["D"]}
        )
        self.assertEqual(self.smiles_set(internal), {"CCOCC", "CCSCC"})
        self.assertEqual(self.smiles_set(hydrogen), {"c1ccccc1"})
        self.assertEqual(self.smiles_set(deuterium), {"[2H]c1ccccc1"})

    def test_frequency_range_uses_one_count_at_every_occurrence(self):
        result = self.instantiator.instantiate(
            "[(CH2)n]O[(CH2)n]", {"n": ["1-2"]}
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["product_count"], 2)
        self.assertEqual(self.smiles_set(result), {"COC", "CCOCC"})

    def test_repeat_count_limit_is_reported_as_truncation(self):
        instantiator = MarkushInstantiator(
            FragmentResolver(fragment_library_path=None),
            InstantiationLimits(max_repeat_count=2),
        )
        result = instantiator.instantiate(
            "[(CH2)n]O[(CH2)n]", {"n": ["1-3"]}
        )

        self.assertEqual(result["product_count"], 2)
        self.assertTrue(result["truncated"])
        self.assertFalse(result["enumeration_complete"])
        self.assertEqual(
            result["frequency"]["values_above_repeat_limit"],
            {"n": [3]},
        )

    def test_sulfur_oxidation_frequency(self):
        result = self.instantiator.instantiate(
            "CS([R1])[(O)m]",
            {"R1": ["methyl"], "m": ["0", "1", "2"]},
        )
        self.assertEqual(
            self.smiles_set(result),
            {"CSC", "CS(C)=O", "CS(C)(=O)=O"},
        )

    def test_position_variation_is_enumerated_but_marked_inferred(self):
        result = self.instantiator.instantiate(
            "Cc1ccc([R1$])nc1", {"R1": ["Cl"]}
        )
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["enumeration_complete"])
        self.assertTrue(result["is_fully_enumerated"])
        self.assertGreater(result["product_count"], 1)
        self.assertTrue(result["position_variation"]["present"])
        self.assertEqual(
            result["position_variation"]["strategy"],
            "inferred_host_ring",
        )
        self.assertIn(
            "position_sites_inferred_from_host_ring", result["warnings"]
        )
        self.assertTrue(
            all(
                product["position_assignment"]
                for product in result["products"]
            )
        )

    def test_position_variation_deduplicates_symmetric_host_ring_sites(self):
        result = self.instantiator.instantiate(
            "Cc1ccccc1[R1$]", {"R1": ["Cl"]}
        )
        detail = result["position_variation"]["reports"][0]["details"][0]

        # Six ring atoms are considered. The methyl-bearing atom cannot take
        # another substituent, and the remaining sites collapse to the three
        # distinct ortho/meta/para products.
        self.assertEqual(detail["candidate_sites"], 6)
        self.assertEqual(detail["chemically_valid_sites"], 5)
        self.assertEqual(detail["valid_unique_sites"], 3)
        self.assertEqual(detail["duplicate_position_variants_removed"], 2)
        self.assertEqual(result["backbone_variants"], 3)
        self.assertEqual(result["product_count"], 3)

    def test_position_variation_does_not_cross_a_fused_ring(self):
        result = self.instantiator.instantiate(
            "c1ccc2ccccc2c1[R1$]", {"R1": ["Cl"]}
        )
        detail = result["position_variation"]["reports"][0]["details"][0]
        host_ring = set(detail["host_ring_atom_indices"])

        self.assertEqual(detail["mode"], "inferred_host_ring")
        self.assertEqual(detail["host_ring_count"], 1)
        self.assertEqual(detail["candidate_sites"], 6)
        self.assertEqual(len(host_ring), 6)
        self.assertTrue(
            all(
                assignment["backbone_atom_index"] in host_ring
                for product in result["products"]
                for assignment in product["position_assignment"]
            )
        )

    def test_ambiguous_host_ring_keeps_the_observed_site(self):
        result = self.instantiator.instantiate(
            "C1CC2([R1$])CCC1C2", {"R1": ["Cl"]}
        )
        detail = result["position_variation"]["reports"][0]["details"][0]

        self.assertEqual(
            result["position_variation"]["strategy"],
            "original_site_only",
        )
        self.assertEqual(detail["mode"], "original_site_only")
        self.assertEqual(detail["reason"], "host_ring_ambiguous")
        self.assertGreater(detail["host_ring_count"], 1)
        self.assertEqual(detail["candidate_sites"], 1)
        self.assertEqual(result["backbone_variants"], 1)
        self.assertEqual(result["product_count"], 1)
        self.assertIn(
            "position_variation_host_ring_ambiguous",
            result["warnings"],
        )

    def test_label_typography_is_aligned_only_to_the_backbone(self):
        result = self.instantiator.instantiate(
            "c1cc([Rc])ccc1[Re]",
            {"R_c": ["Cl"], "R^e": ["methyl"]},
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["product_count"], 1)
        methods = {
            item["method"]
            for item in result["variable_label_alignment"]["mapping"]
        }
        self.assertEqual(methods, {"normalized_typography"})

    def test_uppercase_atom_label_is_not_confused_with_repeat_n(self):
        result = self.instantiator.instantiate(
            "C[(CH2)n][R]", {"N": ["3"], "R": ["methyl"]}
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "frequency_expansion_failed")
        mapping = result["variable_label_alignment"]["mapping"]
        self.assertEqual(mapping[0]["backbone_label"], "N")
        self.assertEqual(mapping[0]["method"], "not_aligned")

    def test_composite_ocsr_placeholder_is_expanded_conservatively(self):
        result = self.instantiator.instantiate(
            "c1cc([OR2])ccc1[R1]",
            {"R1": ["Cl"], "R_2": ["methyl"]},
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["product_count"], 1)
        product = result["products"][0]["smiles"]
        self.assertIn("CO", product)
        expansions = result["core_normalization"][
            "composite_placeholder_expansions"
        ]
        self.assertEqual(expansions[0]["input_token"], "[OR2]")
        self.assertEqual(expansions[0]["expanded_token"], "O[R2]")

    def test_stereochemistry_is_retained(self):
        result = self.instantiator.instantiate(
            "N[C@@H](C)[R1]", {"R1": ["F"]}
        )
        product = result["products"][0]["smiles"]
        mol = Chem.MolFromSmiles(product)
        self.assertIn("@", product)
        self.assertEqual(len(Chem.FindMolChiralCenters(mol)), 1)

    def test_hydrogen_substitution_preserves_a_bracketed_stereocenter(self):
        result = self.instantiator.instantiate(
            "N[C@@]([R1])(C)C(=O)O", {"R1": ["H"]}
        )
        product = result["products"][0]["smiles"]
        mol = Chem.MolFromSmiles(product)
        self.assertEqual(product, "C[C@H](N)C(=O)O")
        self.assertEqual(Chem.FindMolChiralCenters(mol), [(1, "S")])

    def test_two_fragment_grafts_preserve_shared_center_stereochemistry(self):
        core = "N[C@@]([R1])([R2])C(=O)O"
        result = self.instantiator.instantiate(
            core, {"R1": ["methyl"], "R2": ["ethyl"]}
        )
        expected = Chem.MolToSmiles(
            Chem.MolFromSmiles("N[C@@](C)(CC)C(=O)O"),
            isomericSmiles=True,
        )
        self.assertEqual(result["products"][0]["smiles"], expected)

    def test_fragment_graft_preserves_alkene_stereochemistry(self):
        records = [{
            "Description": "E alkenyl",
            "Name": "E alkenyl",
            "SMILE": "[R]/C=C/Cl",
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "fragments.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(records, handle)
            instantiator = MarkushInstantiator(FragmentResolver(path))
            result = instantiator.instantiate(
                "c1ccccc1[R1]", {"R1": ["E alkenyl"]}
            )
        expected = Chem.MolToSmiles(
            Chem.MolFromSmiles("c1ccccc1/C=C/Cl"),
            isomericSmiles=True,
        )
        self.assertEqual(result["products"][0]["smiles"], expected)

    def test_unresolved_values_fail_explicitly(self):
        result = self.instantiator.instantiate(
            "c1ccccc1[R1]", {"R1": ["unknown open class"]}
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["failure_reason"],
            "no_concrete_candidates_for_structural_label:R1",
        )
        self.assertEqual(result["product_count"], 0)

    def test_nonstructural_extracted_variable_is_not_silently_ignored(self):
        result = self.instantiator.instantiate(
            "CC", {"R1": ["methyl"]}
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(self.smiles_set(result), {"CC"})
        self.assertIn(
            "variable_labels_not_present_in_backbone", result["warnings"]
        )

    def test_incompatible_exact_alternative_cannot_report_complete(self):
        result = self.instantiator.instantiate(
            "CC[X]CC", {"X": ["oxygen", "F"]}
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(self.smiles_set(result), {"CCOCC"})
        self.assertEqual(result["invalid_combinations"], 1)
        self.assertIn(
            "chemically_invalid_combinations_skipped", result["warnings"]
        )

    def test_product_limit_is_audited(self):
        instantiator = MarkushInstantiator(
            FragmentResolver(fragment_library_path=None),
            InstantiationLimits(max_products=3),
        )
        result = instantiator.instantiate(
            "[R1]CC[R2]", {"R1": ["halogen"], "R2": ["halogen"]}
        )
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["truncated"])
        self.assertFalse(result["enumeration_complete"])
        self.assertEqual(result["product_count"], 3)
        self.assertEqual(result["theoretical_product_combinations"], 16)

    def test_fragment_candidate_limit_cannot_report_complete(self):
        resolver = FragmentResolver(
            fragment_library_path=None, max_candidates_per_value=2
        )
        result = MarkushInstantiator(resolver).instantiate(
            "c1ccccc1[R1]", {"R1": ["halogen"]}
        )
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["product_count"], 2)
        self.assertIn(
            "fragment_candidate_limit_applied:R1", result["warnings"]
        )

    def test_default_enumeration_has_no_product_cap(self):
        result = self.instantiator.instantiate(
            "[R1]CC[R2]",
            {"R1": ["halogen"], "R2": ["halogen"]},
        )

        self.assertEqual(result["theoretical_product_combinations"], 16)
        self.assertEqual(result["attempted_combinations"], 16)
        self.assertEqual(result["product_count"], 10)
        self.assertEqual(result["duplicate_products_removed"], 6)
        self.assertTrue(result["enumeration_complete"])
        self.assertFalse(result["truncated"])

    def test_auto_audit_below_threshold_writes_txt_and_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            products_path = os.path.join(temp_dir, "products.txt")
            audit_path = os.path.join(temp_dir, "products_audit.jsonl")
            result = self.instantiator.instantiate(
                "c1ccccc1[R1]",
                {"R1": ["halogen"]},
                products_path=products_path,
                audit_path=audit_path,
                audit_mode="auto",
                audit_threshold=4,
            )
            with open(products_path, "r", encoding="utf-8") as handle:
                product_lines = [line.strip() for line in handle if line.strip()]
            with open(audit_path, "r", encoding="utf-8") as handle:
                audit_rows = [json.loads(line) for line in handle if line.strip()]

        self.assertEqual(len(product_lines), 4)
        self.assertEqual(len(audit_rows), 4)
        self.assertEqual(result["products"], [])
        self.assertEqual(result["product_count"], 4)
        self.assertTrue(result["enumeration_complete"])
        self.assertEqual(
            result["output"]["audit_mode_effective"], "detailed"
        )
        self.assertTrue(result["output"]["products_file_written"])
        self.assertTrue(result["output"]["detailed_audit_written"])
        self.assertEqual(len(result["output"]["products_sha256"]), 64)
        self.assertEqual(len(result["output"]["audit_sha256"]), 64)

    def test_auto_audit_above_threshold_writes_only_product_txt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            products_path = os.path.join(temp_dir, "products.txt")
            audit_path = os.path.join(temp_dir, "products_audit.jsonl")
            result = self.instantiator.instantiate(
                "[R1]CC[R2]",
                {"R1": ["halogen"], "R2": ["halogen"]},
                products_path=products_path,
                audit_path=audit_path,
                audit_mode="auto",
                audit_threshold=10,
            )
            self.assertTrue(os.path.isfile(products_path))
            self.assertFalse(os.path.exists(audit_path))

        self.assertEqual(result["theoretical_product_combinations"], 16)
        self.assertEqual(result["product_count"], 10)
        self.assertEqual(
            result["output"]["audit_mode_effective"], "summary_only"
        )
        self.assertFalse(result["output"]["detailed_audit_written"])

    def test_audit_always_and_never_override_the_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            always_products = os.path.join(temp_dir, "always.txt")
            always_audit = os.path.join(temp_dir, "always_audit.jsonl")
            always = self.instantiator.instantiate(
                "c1ccccc1[R1]",
                {"R1": ["halogen"]},
                products_path=always_products,
                audit_path=always_audit,
                audit_mode="always",
                audit_threshold=0,
            )

            never_products = os.path.join(temp_dir, "never.txt")
            never_audit = os.path.join(temp_dir, "never_audit.jsonl")
            never = self.instantiator.instantiate(
                "c1ccccc1[R1]",
                {"R1": ["halogen"]},
                products_path=never_products,
                audit_path=never_audit,
                audit_mode="never",
                audit_threshold=100,
            )

            self.assertTrue(os.path.isfile(always_audit))
            self.assertFalse(os.path.exists(never_audit))

        self.assertEqual(
            always["output"]["audit_mode_effective"], "detailed"
        )
        self.assertEqual(
            never["output"]["audit_mode_effective"], "summary_only"
        )

    def test_overwrite_removes_a_stale_suppressed_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            products_path = os.path.join(temp_dir, "products.txt")
            audit_path = os.path.join(temp_dir, "products_audit.jsonl")
            with open(audit_path, "w", encoding="utf-8") as handle:
                handle.write("stale audit\n")

            result = self.instantiator.instantiate(
                "c1ccccc1[R1]",
                {"R1": ["halogen"]},
                products_path=products_path,
                audit_path=audit_path,
                audit_mode="never",
                overwrite_outputs=True,
            )

            self.assertTrue(os.path.isfile(products_path))
            self.assertFalse(os.path.exists(audit_path))
            self.assertFalse(result["output"]["detailed_audit_written"])

    def test_product_and_audit_paths_must_differ(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "products.txt")
            with self.assertRaisesRegex(ValueError, "must differ"):
                self.instantiator.instantiate(
                    "c1ccccc1[R1]",
                    {"R1": ["halogen"]},
                    products_path=output_path,
                    audit_path=output_path,
                )


if __name__ == "__main__":
    unittest.main()
