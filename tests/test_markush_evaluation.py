import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


MARKUSH_DIR = Path(__file__).resolve().parents[1] / "markush_parsing"
sys.path.insert(0, str(MARKUSH_DIR))

from data_loader import load_combined_label_entries, load_label_entries
from evaluate import compute_aggregate_metrics, markush_graphical_evaluation
from stable_parser import compute_stable_scores


class MarkushGrapherVariableScoringTests(unittest.TestCase):
    def test_standalone_integer_ranges_are_expanded(self):
        scores = compute_stable_scores(
            {"m": ["0-2"], "R1": ["C1-C6 alkyl"]},
            {"m": ["0", "1", "2"], "R1": ["C1-C6 alkyl"]},
        )
        self.assertEqual(scores["variable_precision"], 1.0)
        self.assertEqual(scores["variable_recall"], 1.0)

    def test_aggregate_f1_uses_macro_precision_and_recall(self):
        results = [
            {
                "error": None,
                "scores": {
                    "variable_precision": 1.0,
                    "variable_recall": 0.5,
                },
            },
            {
                "error": None,
                "scores": {
                    "variable_precision": 0.5,
                    "variable_recall": 1.0,
                },
            },
            {
                "error": "preprocessing failed",
                # Even a stale partial score must be ignored after a pipeline
                # failure; the sample contributes zero to system evaluation.
                "scores": {
                    "variable_precision": 1.0,
                    "variable_recall": 1.0,
                },
            },
        ]
        metrics = compute_aggregate_metrics(results)

        # Macro P = Macro R = 0.5 after retaining the failed third sample.
        self.assertEqual(metrics["evaluated"], 3)
        self.assertEqual(metrics["failed_or_unscored_samples"], 1)
        self.assertEqual(metrics["variable_precision_mean"], 0.5)
        self.assertEqual(metrics["variable_recall_mean"], 0.5)
        self.assertEqual(metrics["variable_f1"], 0.5)
        self.assertEqual(
            metrics["variable_f1_from_macro_precision_recall"], 0.5
        )
        self.assertNotIn("variable_f1_mean", metrics)


class PositionVariationTests(unittest.TestCase):
    def test_prediction_in_pseudo_smiles_all_is_correct(self):
        result = markush_graphical_evaluation(
            "CCc1ccc([R3$])nc1",
            "[R3$]C1NCCCC1CC",
            (
                "CCc1ccc([R3])nc1;"
                "[R3]C1NCCCC1CC;"
                "[R3]C1CNCC(CC)C1"
            ),
        )
        self.assertTrue(result["markush_graphical_accuracy"])
        self.assertEqual(result["matched_pseudo_smiles_index"], 1)

    def test_position_variable_label_must_match(self):
        result = markush_graphical_evaluation(
            "CCc1ccc([R3$])nc1",
            "[R2$]C1NCCCC1CC",
            "CCc1ccc([R3])nc1;[R3]C1NCCCC1CC",
        )
        self.assertFalse(result["markush_graphical_accuracy"])
        self.assertEqual(
            result["graphical_evaluation_reason"], "position_variable_mismatch"
        )


class LabelLoadingTests(unittest.TestCase):
    def test_m2s_graphical_and_variable_labels_are_combined(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graphical_path = os.path.join(temp_dir, "graphical.csv")
            variable_path = os.path.join(temp_dir, "labels.json")
            with open(graphical_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image_name",
                        "final_pseudo_smiles",
                        "final_pseudo_smiles_all",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "image_name": "sample.png",
                    "final_pseudo_smiles": "CCc1ccc([R3$])nc1",
                    "final_pseudo_smiles_all": "A;B;C",
                })
            with open(variable_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{
                        "image_name": "sample.png",
                        "annotation": (
                            "<markush><stable>R3:H<n>a halogen atom<ns>m:0-2"
                            "</stable></markush>"
                        ),
                    }],
                    f,
                )

            graphical = load_label_entries(graphical_path)
            combined = load_combined_label_entries(
                graphical_path, variable_path
            )
            self.assertEqual(graphical[0]["pseudo_smiles_all"], ["A", "B", "C"])
            self.assertEqual(
                combined[0]["variables"],
                {"R3": ["H", "a halogen atom"], "m": ["0", "1", "2"]},
            )


if __name__ == "__main__":
    unittest.main()
