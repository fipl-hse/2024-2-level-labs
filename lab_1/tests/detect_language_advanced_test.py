"""
Checks the first lab language detection function
"""

import unittest

import pytest

from lab_1_classify_by_unigrams.main import detect_language_advanced


class DetectLanguageAdvancedTest(unittest.TestCase):
    """
    Tests language detection advanced function
    """

    known_profile = [
        {'name': 'de',
         'freq': {'d': 0.0429, 'e': 0.1453, 'f': 0.0184, 'g': 0.0279, 'a': 0.0653,
                  'b': 0.023, 'c': 0.0356, 'l': 0.039, 'm': 0.0314, 'n': 0.0912,
                  'o': 0.0321, 'h': 0.0512, 'i': 0.0793, 'j': 0.0037, 'k': 0.0167,
                  'u': 0.0385, 't': 0.0636, 'w': 0.0175, 'v': 0.0082, 'q': 0.0003,
                  'p': 0.0113, 's': 0.063, 'r': 0.0666, 'y': 0.0023, 'x': 0.0015,
                  'z': 0.0103, '²': 0.0, 'ä': 0.0039, 'ü': 0.0061, 'ß': 0.0015,
                  'ö': 0.0025, 'é': 0.0}},
        {'name': 'en',
         'freq': {'d': 0.0352, 'e': 0.1106, 'f': 0.02, 'g': 0.0248, 'a': 0.0788,
                  'b': 0.0177, 'c': 0.0252, 'l': 0.0454, 'm': 0.0292, 'n': 0.066,
                  'o': 0.0871, 'h': 0.0476, 'i': 0.0725, 'j': 0.0033, 'k': 0.0135,
                  'u': 0.0325, 't': 0.0855, 'w': 0.025, 'v': 0.0112, 'q': 0.0006,
                  'p': 0.0186, 's': 0.0613, 'r': 0.0542, 'y': 0.0287, 'x': 0.0036,
                  'z': 0.0013, 'é': 0.0, 'ä': 0.0, 'á': 0.0, 'ü': 0.0, 'ö': 0.0}},
        {'name': 'es',
         'freq': {'d': 0.0453, 'e': 0.1312, 'f': 0.0069, 'g': 0.0133, 'a': 0.1251,
                  'b': 0.0133, 'c': 0.0378, 'l': 0.0511, 'm': 0.0321, 'n': 0.0651,
                  'o': 0.0904, 'h': 0.0128, 'i': 0.0555, 'j': 0.0092, 'k': 0.0016,
                  'u': 0.0411, 't': 0.0435, 'w': 0.0012, 'v': 0.0125, 'q': 0.0132,
                  'p': 0.0273, 's': 0.0717, 'r': 0.0608, 'y': 0.0131, 'x': 0.0028,
                  'z': 0.0031, 'ª': 0.0, 'º': 0.0001, 'é': 0.0025, 'á': 0.0035,
                  'ú': 0.0009, 'ñ': 0.0023, 'í': 0.0045, 'è': 0.0, 'ç': 0.0001,
                  'à': 0.0001, 'ü': 0.0001, 'ò': 0.0, 'ó': 0.003}},
        {'name': 'fr',
         'freq': {'d': 0.034, 'e': 0.1424, 'f': 0.0123, 'g': 0.0099, 'a': 0.0822,
                  'b': 0.0115, 'c': 0.0333, 'l': 0.0494, 'm': 0.0329, 'n': 0.0649,
                  'o': 0.0611, 'h': 0.0122, 'i': 0.0711, 'j': 0.0124, 'k': 0.0024,
                  'u': 0.0619, 't': 0.0651, 'w': 0.0021, 'v': 0.0168, 'q': 0.0103,
                  'p': 0.0313, 's': 0.0789, 'r': 0.0649, 'y': 0.0043, 'x': 0.0045,
                  'z': 0.0022, '²': 0.0, 'é': 0.0142, 'ê': 0.0018, 'ç': 0.0018,
                  'à': 0.004, 'ï': 0.0001, 'î': 0.0001, 'ë': 0.0, 'è': 0.0025,
                  'â': 0.0002, 'û': 0.0002, 'ù': 0.0002, 'ô': 0.0004, 'œ': 0.0}}]

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_detect_language_advanced_ideal(self):
        """
        Ideal scenario
        """

        unknown_profile = {
            'name': 'en',
            'freq': {
                'y': 0.0769, 'n': 0.0769, 'e': 0.0769, 'h': 0.1538,
                'm': 0.0769, 'i': 0.0769, 'a': 0.2307, 's': 0.0769,
                'p': 0.1538
            }
        }

        expected = [('es', 0.002), ('fr', 0.0023),
                    ('en', 0.0027), ('de', 0.0028)]
        actual = detect_language_advanced(unknown_profile, DetectLanguageAdvancedTest.known_profile)

        for expected_tuple_with_distance, actual_tuple_with_distance in zip(expected, actual):
            self.assertAlmostEqual(expected_tuple_with_distance[1], actual_tuple_with_distance[1],
                                   delta=1e-4)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_detect_language_advanced_bad_input_unknown_profile(self):
        """
        Bad input scenario
        """

        unknown_profile = ''

        expected = None
        actual = detect_language_advanced(unknown_profile, DetectLanguageAdvancedTest.known_profile)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_detect_language_advanced_bad_input_known_profile(self):
        """
        Bad input scenario
        """

        unknown_profile = {
            'name': 'en',
            'freq': {
                'y': 0.0769, 'n': 0.0769, 'e': 0.0769, 'h': 0.1538,
                'm': 0.0769, 'i': 0.0769, 'a': 0.2307, 's': 0.0769,
                'p': 0.1538
            }
        }

        known_profile = {}

        expected = None
        actual = detect_language_advanced(unknown_profile, known_profile)
        self.assertEqual(expected, actual)
