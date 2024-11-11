"""
Checks the third lab's save_vector function.
"""

import unittest

import pytest

from lab_3_ann_retriever.main import save_vector


class SaveVector(unittest.TestCase):
    """
    Tests vector saving function
    """

    def setUp(self) -> None:
        self.vector = (0.0, -0.007, 0.0, 0.0, 0.5, 0.0, 0.0)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_vector_ideal(self):
        """
        Ideal scenario for saving vector
        """
        expected = {"len": 7, "elements": {1: -0.007, 4: 0.5}}
        actual = save_vector(self.vector)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_vector_return_value(self):
        """
        Checks return type for save_vector function
        """
        actual = save_vector(self.vector)
        self.assertIsInstance(actual, dict)
