"""
Checks the third lab's calculate_distance function.
"""

import unittest

import numpy as np
import pytest

from lab_3_ann_retriever___vhkefbckemvkjo.main import calculate_distance


class CalculateDistance(unittest.TestCase):
    """
    Tests distance calculation function
    """

    def setUp(self) -> None:
        self.query_vector = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0)
        self.document_vector = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_ideal(self):
        """
        Ideal calculate_distance scenario
        """
        expected = 0.8944271909999159
        actual = calculate_distance(self.query_vector, self.document_vector)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_invalid_input(self):
        """
        Invalid input for calculate_distance scenario
        """
        expected = None
        bad_input = None

        actual = calculate_distance(query_vector=bad_input, document_vector=self.document_vector)
        self.assertEqual(expected, actual)

        actual = calculate_distance(query_vector=self.query_vector, document_vector=bad_input)
        self.assertEqual(expected, actual)

        actual = calculate_distance(query_vector=bad_input, document_vector=bad_input)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_return_value(self):
        """
        Checks return type for calculate_distance
        """
        actual = calculate_distance(self.query_vector, self.document_vector)
        self.assertIsInstance(actual, float)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_for_empty_vectors(self):
        """
        Empty vectors scenario for calculate_distance
        """
        actual = calculate_distance((), self.document_vector)
        self.assertEqual(actual, 0.0)
        actual = calculate_distance(self.query_vector, ())
        self.assertEqual(actual, 0.0)
        actual = calculate_distance((), ())
        self.assertEqual(actual, 0.0)
