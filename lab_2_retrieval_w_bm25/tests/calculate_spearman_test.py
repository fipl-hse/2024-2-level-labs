"""
Checks the second lab's Spearman rank correlation calculation function
"""

import unittest

import pytest

from lab_2_retrieval_w_bm25.main import calculate_spearman


class CalculateSpearmanTest(unittest.TestCase):
    """
    Tests for Spearman rank correlation calculation functions
    """

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_calculate_spearman_exact(self):
        """
        Low correlation scenario
        """
        expected = 1.0
        actual = calculate_spearman([9, 0, 3, 7, 4, 1, 2, 6, 5, 8], [9, 0, 3, 7, 4, 1, 2, 6, 5, 8])
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_calculate_spearman_opposite(self):
        """
        Low correlation scenario
        """
        expected = -1.0
        actual = calculate_spearman([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_calculate_spearman_low_negative(self):
        """
        Random ranks correlation scenario
        """
        expected = -0.0424
        actual = calculate_spearman([3, 7, 4, 0, 6, 9, 1, 5, 2, 8], [7, 8, 6, 0, 2, 5, 4, 3, 9, 1])
        self.assertAlmostEqual(expected, actual, delta=1e-4)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_calculate_spearman_high_positive(self):
        """
        Low correlation scenario
        """
        expected = 0.7333
        actual = calculate_spearman([7, 8, 6, 0, 2, 5, 4, 3, 9, 1], [7, 6, 5, 0, 8, 3, 9, 4, 2, 1])
        self.assertAlmostEqual(expected, actual, delta=1e-4)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_calculate_spearman_different_length(self):
        """
        Different length scenario
        """
        expected = None
        actual = calculate_spearman([1, 6, 8, 0], [2, 7, 9])
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_calculate_spearman_bad_input(self):
        """
        Bad input scenario
        """
        expected = None
        bad_list_inputs = [None, True, 42, 3.14, "string", (), {}, [], [1, 3, 5, "seven"], [[]]]

        for bad_input in bad_list_inputs:
            self.assertEqual(calculate_spearman(bad_input, [1, 5, 7]), expected)
            self.assertEqual(calculate_spearman([2, 3, 0], bad_input), expected)
            self.assertEqual(calculate_spearman(bad_input, bad_input), expected)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_calculate_spearman_return_value(self):
        """
        Return value check
        """
        self.assertIsInstance(calculate_spearman([1, 8, 5], [2, 7, 6]), float)
