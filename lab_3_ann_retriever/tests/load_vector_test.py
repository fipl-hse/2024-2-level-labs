"""
Checks the third lab's load_vector function.
"""

import unittest

import pytest

from lab_3_ann_retriever.main import load_vector


class CalculateDistance(unittest.TestCase):
    """
    Tests vector loading function
    """

    def setUp(self) -> None:
        self.state = {"len": 7, "elements": {1: -0.007, 4: 0.5}}

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.skip(reason="rework")
    @pytest.mark.mark10
    def test_load_vector_ideal(self):
        """
        Ideal scenario for loading vector
        """
        expected = (0.0, -0.007, 0.0, 0.0, 0.5, 0.0, 0.0)
        actual = load_vector(self.state)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_vector_missing_elements(self):
        """
        Tests handling a state missing the elements in dictionary.
        """
        state = {"len": 7}
        actual_vector = load_vector(state)
        self.assertIsNone(actual_vector)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_invalid_input_missing_len(self):
        """
        Tests handling a state missing the len in dictionary.
        """
        state = {"elements": {1: -0.007, 4: 0.5}}
        actual_vector = load_vector(state)
        self.assertIsNone(actual_vector)
