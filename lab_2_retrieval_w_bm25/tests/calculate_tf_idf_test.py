"""
Checks the second lab's calculate TF-IDF function
"""

import json
import unittest
from pathlib import Path

import pytest

from lab_2_retrieval_w_bm25.main import calculate_tf_idf


class CalculateTfIdfTest(unittest.TestCase):
    """
    Tests TF-IDF calculation functions
    """
    def setUp(self) -> None:
        with open(Path(__file__).parent / 'test_scores.json', encoding='utf-8') as file:
            scores = json.load(file)

        self.tfs = scores["TF"]

        self.idf = scores["IDF"]

        self.tf_idfs = scores["TF-IDF"]

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_idf_ideal(self):
        """
        Ideal scenario
        """
        expected = self.tf_idfs["Document 9"]

        actual = calculate_tf_idf(self.tfs["Document 9"], self.idf)
        for token, tf_idf in actual.items():
            self.assertAlmostEqual(expected[token], tf_idf, delta=1e-4)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_idf_all_documents(self):
        """
        TF-IDF calculation for all documents
        """
        expected = self.tf_idfs

        for doc_index, tf in self.tfs.items():
            actual = calculate_tf_idf(tf, self.idf)
            for token, token_tf_idf in expected[doc_index].items():
                self.assertAlmostEqual(token_tf_idf, actual[token], delta=1e-4)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_idf_bad_input(self):
        """
        Bad input scenario
        """
        expected = None
        bad_inputs = [None, True, 42, 3.14, (), 'frequency', [],
                      {}, {0: 'bad', 1: 7}, {'token': 'not frequency'}]

        for bad_input in bad_inputs:
            actual = calculate_tf_idf({'good': 0.91, 'tf': 0.54}, bad_input)
            self.assertEqual(expected, actual)

            actual = calculate_tf_idf(bad_input, {'good': 0.91, 'idf': 0.54})
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_idf_return_value(self):
        """
        Function return value check
        """
        actual = calculate_tf_idf(
            {'happy': 0.5, 'token': 0.133},
            {'happy': 1.6, 'token': 1.845}
        )
        self.assertIsInstance(actual, dict)
        for token, tf_idf in actual.items():
            self.assertIsInstance(token, str)
            self.assertIsInstance(tf_idf, float)
