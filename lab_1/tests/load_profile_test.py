"""
Checks the first lab profile loading function
"""
import json
import unittest
from pathlib import Path

import pytest

from lab_1_classify_by_unigrams.main import load_profile


class LoadProfileTest(unittest.TestCase):
    """
    Tests profile loading function
    """

    PATH_TO_LAB_FOLDER = Path(__file__).parent.parent
    PATH_TO_PROFILES_FOLDER = Path(__file__).parent.parent / 'assets' / 'profiles'

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_load_profile_ideal(self):
        """
        Ideal scenario
        """
        path_to_profile = LoadProfileTest.PATH_TO_PROFILES_FOLDER / 'en.json'

        with open(path_to_profile, 'r', encoding='utf-8') as file:
            expected = json.load(file)

        actual = load_profile(str(path_to_profile))
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_load_profile_bad_input_type(self):
        """
        Bad input scenario
        """
        expected = None

        path_to_profile = []
        actual = load_profile(path_to_profile)
        self.assertEqual(expected, actual)
