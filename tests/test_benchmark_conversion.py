"""Regression tests for benchmark representation-conversion utilities."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_DIR = PROJECT_ROOT / "scripts" / "benchmark_conversion"
sys.path.insert(0, str(CONVERTER_DIR))

import convert_cxsmiles_table as cx_table  # noqa: E402
import cxsmiles_to_pseudo_smiles as cx  # noqa: E402
import molfile_to_pseudo_smiles as molfile  # noqa: E402


def _molblock_with_alias(alias: str) -> str:
    block = Chem.MolToMolBlock(Chem.MolFromSmiles("*C"))
    return block.replace("M  END", f"A    1\n{alias}\nM  END")


class CXSMILESConversionTest(unittest.TestCase):
    def test_atom_labels_use_cx_atom_indices(self):
        result = cx.convert_cxsmiles("*C(*)=O |$R1;;R2;$|")
        self.assertEqual(result.status, cx.ConversionStatus.SUCCESS)
        self.assertEqual(result.pseudo_smiles, "[R1]C([R2])=O")

    def test_label_on_non_wildcard_atom_is_preserved(self):
        self.assertEqual(
            cx.cxsmiles_to_pseudo_smiles("C* |$R1;R2$|"), "[R1][R2]"
        )

    def test_explicit_hydrogen_keeps_cx_atom_indexing(self):
        self.assertEqual(
            cx.cxsmiles_to_pseudo_smiles("*C([H])C |$R;;;$|"), "[R]C([H])C"
        )

    def test_single_atom_frequency_group(self):
        result = cx.convert_cxsmiles("COCN1CCOCC1 |Sg:n:2:m:ht:::,|")
        self.assertEqual(result.status, cx.ConversionStatus.SUCCESS)
        self.assertEqual(result.pseudo_smiles, "CO[(CH2)m]N1CCOCC1")

    def test_terminal_frequency_group_requires_review(self):
        result = cx.convert_cxsmiles("OC |Sg:n:1:n:ht|")
        self.assertEqual(result.status, cx.ConversionStatus.REVIEW_REQUIRED)
        self.assertEqual(result.pseudo_smiles, "O[(CH2)n]")
        self.assertIn("TERMINAL_FREQUENCY_GROUP", {i.code for i in result.issues})

    def test_position_variation_is_review_required(self):
        source = "*C.CCC1=CN=CC=C1 |$R3;;;;;;;;;$,m:1:7.8.9.6.4.5|"
        result = cx.convert_cxsmiles(source)
        self.assertEqual(result.status, cx.ConversionStatus.REVIEW_REQUIRED)
        self.assertIn("[R3$]", result.pseudo_smiles or "")
        self.assertEqual(result.metadata["representative_position_targets"], [7])
        with self.assertRaises(cx.CXSMILESToPseudoSMILESError):
            cx.cxsmiles_to_pseudo_smiles(source)

    def test_unlabeled_position_carrier_is_rejected(self):
        result = cx.convert_cxsmiles("*C.C1=CNN=C1 |m:0:3.2.4.5.6|")
        self.assertEqual(result.status, cx.ConversionStatus.UNSUPPORTED)
        self.assertIn(
            "UNLABELED_POSITION_CARRIER", {issue.code for issue in result.issues}
        )

    def test_multi_atom_frequency_group_is_not_guessed(self):
        result = cx.convert_cxsmiles("COCC |Sg:n:1,2:n:ht:::,|")
        self.assertEqual(result.status, cx.ConversionStatus.UNSUPPORTED)
        self.assertIsNone(result.pseudo_smiles)

    def test_unlabeled_wildcard_is_rejected(self):
        result = cx.convert_cxsmiles("*CC")
        self.assertEqual(result.status, cx.ConversionStatus.UNSUPPORTED)
        self.assertIn("UNLABELED_WILDCARD", {issue.code for issue in result.issues})

    def test_unknown_cx_extension_is_rejected(self):
        result = cx.convert_cxsmiles("C=C |c:0|")
        self.assertEqual(result.status, cx.ConversionStatus.UNSUPPORTED)
        self.assertIn(
            "UNSUPPORTED_CX_EXTENSION", {issue.code for issue in result.issues}
        )

    def test_invalid_input_has_invalid_status(self):
        result = cx.convert_cxsmiles("not a smiles")
        self.assertEqual(result.status, cx.ConversionStatus.INVALID)
        self.assertIsNone(result.pseudo_smiles)

    def test_table_conversion_retains_failure_rows_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "predictions.jsonl"
            source.write_text(
                json.dumps({"id": "sample_2", "cxsmiles": "*C |$R1;$|"})
                + "\n"
                + json.dumps({"id": "sample_10", "cxsmiles": None})
                + "\n",
                encoding="utf-8",
            )
            output = root / "converted.csv"
            exit_code = cx_table.main(
                [
                    str(source),
                    str(output),
                    "--image-column",
                    "id",
                    "--output-image-column",
                    "Image_Name",
                    "--image-suffix",
                    ".png",
                    "--pseudo-column",
                    "Predicted_SMILES",
                ]
            )
            self.assertEqual(exit_code, 0)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["id"] for row in rows], ["sample_2", "sample_10"])
        self.assertEqual(rows[0]["Image_Name"], "sample_2.png")
        self.assertEqual(rows[0]["Predicted_SMILES"], "[R1]C")
        self.assertEqual(rows[1]["conversion_status"], "missing_prediction")
        self.assertEqual(rows[1]["Predicted_SMILES"], "")

    def test_single_value_cli_json(self):
        script = CONVERTER_DIR / "cxsmiles_to_pseudo_smiles.py"
        completed = subprocess.run(
            [sys.executable, str(script), "*C(*)=O |$R1;;R2;$|", "--json"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "success")


class MOLFileConversionTest(unittest.TestCase):
    def test_experiment_mode_is_default(self):
        result = molfile.convert_molblock(_molblock_with_alias("R1"))
        self.assertEqual(result.status, molfile.ConversionStatus.SUCCESS)
        self.assertEqual(result.pseudo_smiles, "[R1]C")
        self.assertEqual(result.metadata["conversion_mode"], "experiment")
        self.assertEqual(
            result.metadata["experiment_rdkit_version"],
            molfile.EXPERIMENT_RDKIT_VERSION,
        )

    def test_experiment_mode_normalizes_molscribe_numbered_star(self):
        result = molfile.convert_molblock(_molblock_with_alias("1*"))
        self.assertEqual(result.status, molfile.ConversionStatus.SUCCESS)
        self.assertEqual(result.pseudo_smiles, "[R1]C")

    def test_strict_alias_is_restored_as_pseudo_atom(self):
        result = molfile.convert_molblock(
            _molblock_with_alias("R1"), mode=molfile.ConversionMode.STRICT
        )
        self.assertEqual(result.status, molfile.ConversionStatus.SUCCESS)
        self.assertEqual(result.pseudo_smiles, "C[R1]")
        self.assertEqual(result.metadata["conversion_mode"], "strict")

    def test_strict_molscribe_numbered_star_alias_is_normalized(self):
        result = molfile.convert_molblock(
            _molblock_with_alias("1*"), mode=molfile.ConversionMode.STRICT
        )
        self.assertEqual(result.status, molfile.ConversionStatus.SUCCESS)
        self.assertEqual(result.pseudo_smiles, "C[R1]")

    def test_unlabeled_dummy_is_not_guessed(self):
        block = Chem.MolToMolBlock(Chem.MolFromSmiles("*C"))
        lines = block.splitlines()
        counts_index = next(i for i, line in enumerate(lines) if "V2000" in line)
        atom_line = lines[counts_index + 1]
        lines[counts_index + 1] = atom_line[:31] + "*  " + atom_line[34:]
        block = "\n".join(lines) + "\n"
        result = molfile.convert_molblock(
            block, mode=molfile.ConversionMode.STRICT
        )
        self.assertEqual(result.status, molfile.ConversionStatus.UNSUPPORTED)
        self.assertIn("UNLABELED_DUMMY_ATOM", {issue.code for issue in result.issues})

    def test_standard_molecule_stereochemistry_is_preserved(self):
        source = Chem.MolFromSmiles("F[C@H](Cl)Br")
        block = Chem.MolToMolBlock(source)
        result = molfile.convert_molblock(
            block, mode=molfile.ConversionMode.STRICT
        )
        self.assertEqual(result.status, molfile.ConversionStatus.SUCCESS)
        converted = Chem.MolFromSmiles(result.pseudo_smiles or "")
        self.assertIsNotNone(converted)
        self.assertEqual(
            Chem.FindMolChiralCenters(source, includeUnassigned=True),
            Chem.FindMolChiralCenters(converted, includeUnassigned=True),
        )

    def test_v3000_is_explicitly_unsupported(self):
        block = Chem.MolToV3KMolBlock(Chem.MolFromSmiles("CC"))
        result = molfile.convert_molblock(
            block, mode=molfile.ConversionMode.STRICT
        )
        self.assertEqual(result.status, molfile.ConversionStatus.UNSUPPORTED)
        self.assertIn(
            "UNSUPPORTED_MOLFILE_VERSION", {issue.code for issue in result.issues}
        )

    def test_directory_batch_keeps_invalid_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "valid.mol").write_text(
                _molblock_with_alias("R2"), encoding="utf-8"
            )
            (root / "failed.mol").write_text("not a mol block\n", encoding="utf-8")
            output = root / "converted.csv"
            exit_code = molfile.main(
                [str(root), "--output", str(output), "--mode", "strict"]
            )
            self.assertEqual(exit_code, 0)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = {row["Image_Name"]: row for row in csv.DictReader(handle)}
        self.assertEqual(set(rows), {"failed.png", "valid.png"})
        self.assertEqual(rows["valid.png"]["Predicted_SMILES"], "C[R2]")
        self.assertEqual(rows["failed.png"]["conversion_status"], "invalid")
        self.assertEqual(rows["failed.png"]["Predicted_SMILES"], "")


if __name__ == "__main__":
    unittest.main()
