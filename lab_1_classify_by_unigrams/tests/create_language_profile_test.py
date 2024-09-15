"""
Checks the first lab language profile creation function
"""

import unittest

import pytest

from lab_1_classify_by_unigrams.main import create_language_profile


class CreateLanguageProfileTest(unittest.TestCase):
    """
    Tests language profile creation function
    """

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_create_profile_ideal(self):
        """
        Ideal scenario
        """
        expected = {'name': 'en',
                    'freq': {
                        'y': 0.0769, 'n': 0.0769, 'e': 0.0769, 'h': 0.1538, 'm': 0.0769,
                        'i': 0.0769, 'a': 0.2307, 's': 0.0769, 'p': 0.1538}}
        language_name = 'en'
        text = 'he is a happy man'
        actual = create_language_profile(language_name, text)
        self.assertEqual(expected['name'], actual['name'])

        for tuple_with_frequencies in expected['freq'].items():
            frequencies = actual['freq'][tuple_with_frequencies[0]]
            self.assertAlmostEqual(tuple_with_frequencies[1], frequencies, delta=1e-4)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_create_profile_deutsch_ideal(self):
        """
        Ideal scenario with deutsch
        """
        expected = {'name': 'de',
                    'freq': {
                        'n': 0.0666, 's': 0.0333, 'a': 0.0666, 'm': 0.0666, 't': 0.0666,
                        'i': 0.1333, 'w': 0.0666, 'ß': 0.0333, 'ö': 0.0333, 'e': 0.1,
                        'h': 0.1666, 'c': 0.1666}}
        language_name = 'de'
        text = 'Ich weiß nicht was ich machen möchte.'
        actual = create_language_profile(language_name, text)
        self.assertEqual(expected['name'], actual['name'])

        for tuple_with_frequencies in expected['freq'].items():
            frequencies = actual['freq'][tuple_with_frequencies[0]]
            self.assertAlmostEqual(tuple_with_frequencies[1], frequencies, delta=1e-4)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_create_profile_bad_input(self):
        """
        Bad input scenario
        """
        expected = None
        language_name = 'de'
        text = []
        actual = create_language_profile(language_name, text)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_create_profile_bad_input_lang_name(self):
        """
        Bad input scenario, language name
        """
        expected = None
        language_name = 123
        text = 'Ich weiß nicht was ich machen möchte. Vielleicht ich muss das überlegen'
        actual = create_language_profile(language_name, text)
        self.assertEqual(expected, actual)
