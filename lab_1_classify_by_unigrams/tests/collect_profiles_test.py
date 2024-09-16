"""
Checks the first lab language profile collection function
"""

import unittest
from pathlib import Path

import pytest

from lab_1_classify_by_unigrams.main import collect_profiles


class CollectProfilesTest(unittest.TestCase):
    """
    Tests profile collection function
    """

    PATH_TO_PROFILES_FOLDER = Path(__file__).parent.parent / 'assets' / 'profiles'

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_collect_profiles_ideal(self):
        """
        Ideal scenario
        """

        expected = [
            {'name': 'it',
             'freq': {'d': 0.034, 'e': 0.104, 'f': 0.0119, 'g': 0.0183, 'a': 0.1133,
                      'b': 0.012, 'c': 0.0461, 'l': 0.0549, 'm': 0.0323, 'n': 0.07,
                      'o': 0.1005, 'h': 0.0171, 'i': 0.1063, 'j': 0.0007, 'k': 0.0019,
                      'u': 0.0298, 't': 0.0619, 'w': 0.002, 'v': 0.0176, 'q': 0.0045,
                      'p': 0.0293, 's': 0.0509, 'r': 0.0602, 'y': 0.0015, 'x': 0.0013,
                      'z': 0.0082, 'è': 0.0041, 'í': 0.0, 'ì': 0.0008, 'é': 0.0005, 'ç': 0.0004,
                      'á': 0.0, 'à': 0.0016, 'ú': 0.0, 'ù': 0.0011, 'ò': 0.0009, 'ó': 0.0}},
            {'name': 'ru',
             'freq': {'ь': 0.0194, 'э': 0.003, 'ю': 0.0064, 'я': 0.0215, 'ш': 0.0077,
                      'щ': 0.0031, 'ъ': 0.0002, 'ы': 0.0171, 'ф': 0.004, 'х': 0.0096,
                      'ц': 0.0046, 'ч': 0.0149, 'р': 0.0499, 'с': 0.0523, 'т': 0.066,
                      'у': 0.0288, 'ё': 0.0009, 'й': 0.0125, 'и': 0.0705, 'л': 0.0399,
                      'к': 0.0422, 'н': 0.063, 'м': 0.0306, 'п': 0.0284, 'о': 0.1,
                      'б': 0.0189, 'а': 0.0854, 'г': 0.0161, 'в': 0.0418, 'е': 0.0829,
                      'д': 0.0314, 'з': 0.0166, 'ж': 0.0097, '가': 0.0}},
            {'name': 'tr',
             'freq': {'d': 0.0435, 'e': 0.0928, 'f': 0.0066, 'g': 0.0156, 'a': 0.1211,
                      'b': 0.0282, 'c': 0.0129, 'l': 0.0575, 'm': 0.047, 'n': 0.0705,
                      'o': 0.0325, 'h': 0.0137, 'i': 0.0868, 'j': 0.0008, 'k': 0.0441,
                      'u': 0.0343, 't': 0.0331, 'w': 0.0009, 'v': 0.0096, 'q': 0.0001,
                      'p': 0.0093, 's': 0.0381, 'r': 0.062, 'y': 0.0381, 'x': 0.0002,
                      'z': 0.0176, 'ç': 0.0081, 'ü': 0.0126, 'ß': 0.0, 'ö': 0.0052,
                      'î': 0.0, 'é': 0.0, 'â': 0.0001, 'û': 0.0, 'ğ': 0.0062, 'ı': 0.0372,
                      'i̇': 0.0011, 'ş': 0.0122}}]

        paths_to_profiles = [
            str(CollectProfilesTest.PATH_TO_PROFILES_FOLDER / 'it.json'),
            str(CollectProfilesTest.PATH_TO_PROFILES_FOLDER / 'ru.json'),
            str(CollectProfilesTest.PATH_TO_PROFILES_FOLDER / 'tr.json')]

        actual = collect_profiles(paths_to_profiles)
        for expected_dictionary_with_profiles, actual_dictionary_with_profiles \
                in zip(expected, actual):
            self.assertEqual(expected_dictionary_with_profiles['name'],
                             actual_dictionary_with_profiles['name'])

            for tuple_with_frequencies in expected_dictionary_with_profiles['freq'].items():
                frequencies = actual_dictionary_with_profiles['freq'][tuple_with_frequencies[0]]
                self.assertAlmostEqual(tuple_with_frequencies[1], frequencies, delta=1e-4)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_preprocess_profile_bad_input_type(self):
        """
        Bad input scenario
        """
        expected = None

        bad_inputs = ['string', {}, (), None, 9, 9.34, True]
        for bad_input in bad_inputs:
            actual = collect_profiles(bad_input)
            self.assertEqual(expected, actual)
