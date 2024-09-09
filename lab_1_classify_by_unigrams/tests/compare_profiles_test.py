"""
Checks the first lab language comparison function
"""

import unittest

import pytest

from lab_1_classify_by_unigrams.main import compare_profiles


class CompareProfilesTest(unittest.TestCase):
    """
    Tests profile comparison function
    """

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_compare_profiles_ideal(self):
        """
        Ideal scenario
        """
        unknown_profile = {
            'name': 'en',
            'freq': {
                'y': 0.0769, 'n': 0.0769, 'e': 0.0769, 'h': 0.1538, 'm': 0.0769, 'i': 0.0769,
                'a': 0.2307, 's': 0.0769, 'p': 0.1538
            }
        }

        profile_to_compare = {
            'name': 'de',
            'freq': {
                'u': 0.0344, 'e': 0.1724, 'a': 0.0344, 'r': 0.0344, 'i': 0.1034, 'g': 0.0344,
                'b': 0.0344, 't': 0.0344, 'd': 0.0344, 'm': 0.0344, 's': 0.1034, 'h': 0.0689,
                'c': 0.0689, 'v': 0.0344, 'l': 0.1034, 'ü': 0.0344, 'n': 0.0344
            }
        }

        expected = 0.006
        actual = compare_profiles(unknown_profile, profile_to_compare)
        self.assertAlmostEqual(expected, actual, delta=1e-4)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_compare_profiles_no_intersections_ideal(self):
        """
        Ideal scenario with no intersections
        """
        unknown_profile = {
            'name': 'en',
            'freq': {
                'a': 1.0
            }
        }

        profile_to_compare = {
            'name': 'de',
            'freq': {
                'ß': 1.0
            }
        }

        expected = 1.0
        actual = compare_profiles(unknown_profile, profile_to_compare)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_compare_profiles_identical(self):
        """
        Ideal scenario with identical profiles
        """
        unknown_profile = {
            'name': 'en',
            'freq': {
                'p': 0.25, 'y': 0.125, 'h': 0.125,
                'a': 0.25, 'm': 0.125, 'n': 0.125
            }
        }

        profile_to_compare = {
            'name': 'de',
            'freq': {
                'p': 0.25, 'y': 0.125, 'h': 0.125,
                'a': 0.25, 'm': 0.125, 'n': 0.125
            }
        }

        expected = 0.0
        actual = compare_profiles(unknown_profile, profile_to_compare)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_compare_profiles_bad_input_type(self):
        """
        Bad input scenario
        """
        unknown_profile = []

        profile_to_compare = {'name': 'de',
                      'freq': {
                          'n': 0.0666, 's': 0.0333, 'a': 0.0666, 'm': 0.0666,
                          't': 0.0666, 'i': 0.1333, 'w': 0.0666, 'ß': 0.0333,
                          'ö': 0.0333, 'e': 0.1, 'h': 0.1666, 'c': 0.1666}}

        expected = None
        actual = compare_profiles(unknown_profile, profile_to_compare)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_compare_profiles_bad_input_profile(self):
        """
        Bad input scenario
        """
        unknown_profile = {
            'freq': {
                'y': 0.0769, 'n': 0.0769, 'e': 0.0769, 'h': 0.1538, 'm': 0.0769, 'i': 0.0769,
                'a': 0.2307, 's': 0.0769, 'p': 0.1538
            }
        }

        profile_to_compare = {
            'name': 'de',
            'freq': {
                'u': 0.0344, 'e': 0.1724, 'a': 0.0344, 'r': 0.0344, 'i': 0.1034, 'g': 0.0344,
                'b': 0.0344, 't': 0.0344, 'd': 0.0344, 'm': 0.0344, 's': 0.1034, 'h': 0.0689,
                'c': 0.0689, 'v': 0.0344, 'l': 0.1034, 'ü': 0.0344, 'n': 0.0344
            }
        }

        expected = None
        actual = compare_profiles(unknown_profile, profile_to_compare)
        self.assertEqual(expected, actual)
