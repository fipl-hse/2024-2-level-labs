# pylint: disable=duplicate-code
"""
Checks the second lab's index saving function
"""
import unittest
from pathlib import Path

import pytest

from lab_2_retrieval_w_bm25.main import load_index


class SaveIndexTest(unittest.TestCase):
    """
    Tests for index saving functions
    """
    def setUp(self) -> None:
        self.metrics_file_path = str(Path(__file__).parent / "saved_metrics_example.json")

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_load_index_ideal(self):
        """
        Ideal scenario
        """
        expected = {
            'string': 0.0, 'ghost': 0.0, 'named': 0.0, 'saw': 0.5526,
            'beautiful': 0.3664, 'particular': 0.0, 'reading': 0.5526,
            'library': 0.5522, 'kind': 0.5526, 'opened': 0.0, 'boy': 0.5526,
            'street': 0.3664, 'must': 0.0, 'need': 0.0, 'lived': 0.5526,
            'smart': 0.5526, 'began': 0.0, 'friend': 0.9871, 'coffee': 0.9871,
            'upon': 0.5526, 'time': 0.5526, 'love': 0.0, 'writing': 0.0,
            'format': 0.0, 'story': 0.0, 'cat': 0.4075, 'function': 0.0,
            'loved': 0.9871, 'tale': 0.0, 'told': 0.0, 'thought': 0.0,
            'shop': 0.9871, 'ugly': 0.0, 'output': 0.0, 'across': 0.3664
        }

        actual = load_index(self.metrics_file_path)
        for token, score in expected.items():
            self.assertEqual(score, actual[0][token])

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_load_index_bad_input(self):
        """
        Bad input scenario
        """
        expected = None
        bad_path_inputs = [None, True, 42, 3.14, (), {}, [], '']

        for bad_path in bad_path_inputs:
            actual = load_index(bad_path)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_save_index_return_value(self):
        """
        Return value check
        """
        actual = load_index(self.metrics_file_path)

        self.assertIsInstance(actual, list)
        for doc_metrics in actual:
            self.assertIsInstance(doc_metrics, dict)
            for token, score in doc_metrics.items():
                self.assertIsInstance(token, str)
                self.assertIsInstance(score, float)
