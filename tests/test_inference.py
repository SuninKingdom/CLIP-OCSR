"""Tests for CLIP-OCSR inference pipeline."""

import unittest
import torch
import numpy as np
from PIL import Image
import tempfile
import os


class TestPreprocessing(unittest.TestCase):
    """Test image preprocessing."""

    def test_preprocess_image(self):
        from clip_ocsr.inference.predict import preprocess_image

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (256, 256), color=(255, 255, 255))
            img.save(f.name)
            temp_path = f.name

        try:
            tensor = preprocess_image(temp_path)
            self.assertEqual(tensor.shape, (1, 3, 512, 512))
            self.assertTrue(tensor.min() >= 0)
            self.assertTrue(tensor.max() <= 1)
        finally:
            os.unlink(temp_path)


class TestAbbrevGroup(unittest.TestCase):
    """Test abbreviation group expansion."""

    def test_abbrevgroup2smiles_no_abbrev(self):
        from clip_ocsr.utils.abbrev_group import abbrevgroup2smiles
        smiles = "CCO"
        result = abbrevgroup2smiles(smiles, "assets/abbrev_group.json")
        self.assertEqual(result, "CCO")

    def test_abbrevgroup2smiles_with_abbrev(self):
        from clip_ocsr.utils.abbrev_group import abbrevgroup2smiles
        # [Me] should expand to C
        smiles = "C[Me]"
        result = abbrevgroup2smiles(smiles, "assets/abbrev_group.json")
        self.assertIn("C", result)
        self.assertNotIn("[Me]", result)


class TestSeed(unittest.TestCase):
    """Test seed utility."""

    def test_set_seed(self):
        from clip_ocsr.utils.seed import set_seed
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
