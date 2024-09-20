"""
Checks the first lab calculate frequencies function
"""

import unittest

import pytest

from lab_1_classify_by_unigrams.main import calculate_frequencies


class CalculateFrequenciesTest(unittest.TestCase):
    """
    Tests calculating frequencies function
    """

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequencies_ideal(self):
        """
        Ideal calculate frequencies scenario
        """
        expected = {'a': 0.1666, 'n': 0.1666,
                    'm': 0.1666, 'h': 0.1666,
                    't': 0.1666, 'e': 0.1666}

        actual = calculate_frequencies(['t', 'h', 'e', 'm', 'a', 'n'])

        self.assertEqual(expected.keys(), actual.keys())

        for tuple_with_frequencies in expected.items():
            frequencies = actual[tuple_with_frequencies[0]]
            self.assertAlmostEqual(tuple_with_frequencies[1], frequencies, delta=1e-4)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequencies_complex(self):
        """
        Calculate frequencies with several same tokens
        """
        expected = {'t': 0.1176, 'r': 0.0588, 'u': 0.0588, 'n': 0.1176,
                      'a': 0.0588, 'y': 0.0588, 'h': 0.1176, 'i': 0.0588,
                      's': 0.1176, 'w': 0.0588, 'e': 0.1764}
        actual = calculate_frequencies(['t', 'h', 'e', 'w', 'e', 'a', 't', 'h', 'e',
                                          'r', 'i', 's', 's', 'u', 'n', 'n', 'y'])

        self.assertEqual(expected.keys(), actual.keys())

        for tuple_with_frequencies in expected.items():
            frequencies = actual[tuple_with_frequencies[0]]
            self.assertAlmostEqual(tuple_with_frequencies[1], frequencies, delta=1e-4)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequencies_bad_input(self):
        """
        Calculate frequencies invalid input tokens check
        """
        bad_inputs = ['string', {}, (), None, 9, 9.34, True, [None]]
        expected = None
        for bad_input in bad_inputs:
            actual = calculate_frequencies(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequencies_return_value(self):
        """
        Calculate frequencies return values check
        """
        tokens = ['t', 'h', 'e', 'm', 'a', 'n']
        expected = 6
        actual = calculate_frequencies(tokens)
        self.assertEqual(expected, len(actual))
        for token in tokens:
            self.assertTrue(actual[token])
        self.assertTrue(isinstance(actual[tokens[0]], float))
