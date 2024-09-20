"""
Checks the first lab language detection function
"""

import unittest

import pytest

from lab_1_classify_by_unigrams.main import detect_language


class DetectLanguageTest(unittest.TestCase):
    """
    Tests language detection function
    """

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_detect_language_ideal(self):
        """
        Ideal scenario
        """

        unknown_profile = {
            'name': 'unk',
            'freq': {
                'm': 0.0909, 'e': 0.0909, 'h': 0.1818, 'p': 0.1818,
                'y': 0.0909, 's': 0.0909, 'n': 0.0909, 'a': 0.1818
            }
        }

        en_profile = {
            'name': 'en',
            'freq': {
                'p': 0.2, 'y': 0.1, 'e': 0.1, 'h': 0.2,
                'a': 0.2, 'm': 0.1, 'n': 0.1
            }
        }

        de_profile = {'name': 'de',
                      'freq': {
                          'n': 0.0666, 's': 0.0333, 'a': 0.0666, 'm': 0.0666,
                          't': 0.0666, 'i': 0.1333, 'w': 0.0666, 'ß': 0.0333,
                          'ö': 0.0333, 'e': 0.1, 'h': 0.1666, 'c': 0.1666}}

        expected = en_profile['name']
        actual = detect_language(unknown_profile, en_profile, de_profile)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_detect_language_deutsch_ideal(self):
        """
        Ideal scenario with deutsch
        """

        unknown_profile = {
            'name': 'unk',
            'freq': {
                'e': 0.2222, 'h': 0.0555, 'c': 0.0555, 't': 0.0555,
                'r': 0.0555, 'g': 0.0555, 'b': 0.0555, 'w': 0.0555,
                'l': 0.0555, 'ß': 0.0555, 'i': 0.1111, 'ü': 0.0555, 'n': 0.1111
            }
        }

        en_profile = {
            'name': 'en',
            'freq': {
                'p': 0.2, 'y': 0.1, 'e': 0.1, 'h': 0.2,
                'a': 0.2, 'm': 0.1, 'n': 0.1
            }
        }

        de_profile = {
            'name': 'de',
            'freq': {
                't': 0.0833, 'h': 0.1666, 'n': 0.0833, 'w': 0.0833,
                'ß': 0.0833, 'e': 0.0833, 'c': 0.1666, 'i': 0.25
            }
        }

        expected = de_profile['name']
        actual = detect_language(unknown_profile, en_profile, de_profile)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_detect_language_bad_input(self):
        """
        Bad input scenario
        """

        unknown_profile = []

        en_profile = {
            'name': 'en',
            'freq': {
                'p': 0.2, 'y': 0.1, 'e': 0.1, 'h': 0.2,
                'a': 0.2, 'm': 0.1, 'n': 0.1
            }
        }

        de_profile = {
            'name': 'de',
            'freq': {
                't': 0.0833, 'h': 0.1666, 'n': 0.0833, 'w': 0.0833,
                'ß': 0.0833, 'e': 0.0833, 'c': 0.1666, 'i': 0.25
            }
        }

        expected = None
        actual = detect_language(unknown_profile, en_profile, de_profile)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_detect_language_bad_input_profile(self):
        """
        Bad input scenario
        """

        unknown_profile = {
            'name': 'unk',
            'freq': {
                'm': 0.0909, 'e': 0.0909, 'h': 0.1818, 'p': 0.1818,
                'y': 0.0909, 's': 0.0909, 'n': 0.0909, 'a': 0.1818
            }
        }

        en_profile = []

        de_profile = {
            'name': 'de',
            'freq': {
                't': 0.0833, 'h': 0.1666, 'n': 0.0833, 'w': 0.0833,
                'ß': 0.0833, 'e': 0.0833, 'c': 0.1666, 'i': 0.25
            }
        }

        expected = None
        actual = detect_language(unknown_profile, en_profile, de_profile)
        self.assertEqual(expected, actual)
