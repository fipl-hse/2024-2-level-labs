"""
Checks the first lab calculation of the mean squared error function
"""

import unittest

import pytest

from lab_1_classify_by_unigrams.main import calculate_mse


class CalculateMSETest(unittest.TestCase):
    """
    Tests calculation of the MSE function
    """

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_mse_ideal(self):
        """
        Ideal scenario
        """
        predicted_value = [0.1538, 0.0, 0.0, 0.0769, 0.0769, 0.0769, 0.0, 0.0,
                           0.0769, 0.0769, 0.0769, 0.1538, 0.2307, 0.0]

        actual_value = [0.1666, 0.1666, 0.0333, 0.1333, 0.0, 0.0666, 0.0666,
                        0.0333, 0.0333, 0.1, 0.0666, 0.0, 0.0666, 0.0666]

        expected = 0.0072
        actual = calculate_mse(predicted_value, actual_value)
        self.assertAlmostEqual(expected, actual, delta=1e-4)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_mse_bad_input_length(self):
        """
        Bad input scenario
        """
        predicted_value = [0.1538, 0.0, 0.0, 0.0769, 0.0769, 0.0769, 0.0, 0.0,
                           0.0769, 0.0769, 0.0769, 0.1538, 0.2307, 0.0]

        actual_value = [0.1666, 0.1666, 0.0333, 0.1333, 0.0, 0.0666, 0.0666,
                        0.0333, 0.0333, 0.1]

        expected = None
        actual = calculate_mse(predicted_value, actual_value)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_mse_bad_input_predicted(self):
        """
        Bad input scenario
        """
        predicted_value = '0.0'

        actual_value = [0.1666, 0.1666, 0.0333, 0.1333, 0.0, 0.0666, 0.0666,
                        0.0333, 0.0333, 0.1, 0.0666, 0.0, 0.0666, 0.0666]

        expected = None
        actual = calculate_mse(predicted_value, actual_value)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_mse_bad_input_actual(self):
        """
        Bad input scenario
        """
        predicted_value = [0.1538, 0.0, 0.0, 0.0769, 0.0769, 0.0769, 0.0, 0.0,
                           0.0769, 0.0769, 0.0769, 0.1538, 0.2307, 0.0]

        actual_value = {}

        expected = None
        actual = calculate_mse(predicted_value, actual_value)
        self.assertEqual(expected, actual)
