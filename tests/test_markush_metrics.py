import csv
import tempfile
import unittest
from pathlib import Path

from clip_ocsr.evaluation.markush_metrics import (
    evaluate_prediction_files,
    markush_graphical_evaluation,
)


class MarkushGraphicalMetricTests(unittest.TestCase):
    def test_equivalent_pseudo_smiles_are_compared_as_graphs(self):
        result = markush_graphical_evaluation("[R1]CC", "CC[R1]")
        self.assertTrue(result["markush_graphical_accuracy"])
        self.assertEqual(
            result["graphical_evaluation_reason"], "canonical_graph_match"
        )

    def test_position_variation_uses_reviewed_candidate_set(self):
        result = markush_graphical_evaluation(
            "CCc1ccc([R3$])nc1",
            "[R3$]C1NCCCC1CC",
            "CCc1ccc([R3])nc1;[R3]C1NCCCC1CC",
        )
        self.assertTrue(result["markush_graphical_accuracy"])
        self.assertEqual(result["matched_pseudo_smiles_index"], 1)

    def test_tetrahedral_stereoisomers_are_not_equal(self):
        result = markush_graphical_evaluation(
            "O[C@H](C)c1ccccc1", "O[C@@H](C)c1ccccc1"
        )
        self.assertFalse(result["markush_graphical_accuracy"])

    def test_alkene_stereoisomers_are_not_equal(self):
        result = markush_graphical_evaluation("F/C=C/F", "F/C=C\\F")
        self.assertFalse(result["markush_graphical_accuracy"])


class MarkushBatchEvaluationTests(unittest.TestCase):
    def test_all_label_rows_remain_in_the_denominator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            labels_path = Path(temp_dir) / "labels.csv"
            predictions_path = Path(temp_dir) / "predictions.csv"

            with labels_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_name",
                        "final_pseudo_smiles",
                        "final_pseudo_smiles_all",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "image_name": "a.png",
                            "final_pseudo_smiles": "[R1]CC",
                            "final_pseudo_smiles_all": "[R1]CC",
                        },
                        {
                            "image_name": "b.png",
                            "final_pseudo_smiles": "CCc1ccc([R3$])nc1",
                            "final_pseudo_smiles_all": (
                                "CCc1ccc([R3])nc1;[R3]C1NCCCC1CC"
                            ),
                        },
                        {
                            "image_name": "c.png",
                            "final_pseudo_smiles": "[R2]O",
                            "final_pseudo_smiles_all": "[R2]O",
                        },
                    ]
                )

            with predictions_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["Image_Name", "Predicted_SMILES"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"Image_Name": "a.png", "Predicted_SMILES": "CC[R1]"},
                        {
                            "Image_Name": "b.png",
                            "Predicted_SMILES": "[R3$]C1NCCCC1CC",
                        },
                    ]
                )

            summary, details = evaluate_prediction_files(
                labels_path, predictions_path
            )

            self.assertEqual(summary["total_samples"], 3)
            self.assertEqual(summary["correct_samples"], 2)
            self.assertEqual(summary["missing_predictions"], 1)
            self.assertAlmostEqual(summary["markush_graphical_accuracy"], 2 / 3)
            self.assertEqual(len(details), 3)
            self.assertEqual(
                details[-1]["graphical_evaluation_reason"], "missing_prediction"
            )


if __name__ == "__main__":
    unittest.main()
