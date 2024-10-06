"""
Checks the second lab's calculate TF function
"""

import unittest

import pytest

from lab_2_retrieval_w_bm25.main import calculate_tf


class CalculateTfTest(unittest.TestCase):
    """
    Tests TF calculation functions
    """
    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_familiar_tokens(self):
        """
        Scenario with tokens already in vocabulary
        """
        vocabulary = [
            'upon', 'time', 'tale', 'boy', 'smart', 'kind', 'stories', 'need',
            'told', 'story', 'began', 'coffee', 'shop', 'across', 'street',
            'love', 'reading', 'writing', 'friend', 'opened', 'library', 'named',
            'function', 'must', 'output', 'string', 'particular', 'format',
            'beautiful', 'ugly', 'thought', 'saw', 'ghost', 'cat', 'lived', 'loved'
        ]

        expected = {
            'saw': 0.0, 'ugly': 0.0, 'friend': 0.0, 'output': 0.0, 'lived': 0.0,
            'time': 0.0, 'must': 0.0, 'love': 0.0, 'reading': 0.0, 'upon': 0.0,
            'kind': 0.0, 'smart': 0.0, 'cat': 0.3333, 'boy': 0.0, 'shop': 0.0,
            'began': 0.0, 'beautiful': 0.1667, 'thought': 0.0, 'street': 0.0,
            'stories': 0.1667, 'library': 0.1667, 'format': 0.0, 'loved': 0.1667,
            'writing': 0.0, 'told': 0.0, 'named': 0.0, 'particular': 0.0, 'need': 0.0,
            'across': 0.0, 'function': 0.0, 'string': 0.0, 'ghost': 0.0, 'coffee': 0.0,
            'story': 0.0, 'tale': 0.0, 'opened': 0.0
        }

        document_tokens = ['library', 'cat', 'loved', 'stories', 'beautiful', 'cat']
        actual = calculate_tf(vocabulary, document_tokens)

        for token, frequency in actual.items():
            self.assertAlmostEqual(expected[token], frequency, delta=1e-4)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_new_tokens(self):
        """
        Scenario with tokens that are not in vocabulary
        """
        expected_vocabulary = {
            'simple': 0.0, 'vocabulary': 0.0, 'document': 0.3333,
            'none': 0.3333, 'words': 0.3333
        }
        actual = calculate_tf(['simple', 'vocabulary'], ['document', 'none', 'words'])
        for token, frequency in expected_vocabulary.items():
            self.assertAlmostEqual(frequency, actual[token], delta=1e-4)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_mixed_tokens(self):
        """
        Scenario for mixed types of tokens
        """
        vocabulary = [
            'time', 'ghosts', 'opened', 'format', 'stories', 'boy', 'make', 'need',
            'cat', 'lived', 'beautiful', 'police', 'taken', 'tale', 'like', 'upon',
            'ghost', 'kid', 'house', 'friend', 'string', 'ugly', 'case', 'spooky',
            'lots', 'unsure', 'smart', 'thought', 'loved', 'real', 'cats', 'library'
        ]

        expected = {
            'speak': 0.04, 'real': 0.0, 'okay': 0.04, 'boy': 0.0, 'string': 0.0,
            'format': 0.0, 'tale': 0.0, 'beautiful': 0.0, 'still': 0.04, 'talent': 0.04,
            'could': 0.04, 'upon': 0.0, 'opened': 0.0, 'sister': 0.04, 'unusual': 0.08,
            'told': 0.04, 'scared': 0.04, 'need': 0.0, 'day': 0.04, 'time': 0.0,
            'taken': 0.0, 'police': 0.0, 'cats': 0.08, 'friend': 0.0, 'house': 0.0,
            'love': 0.04, 'case': 0.0, 'one': 0.04, 'unsure': 0.0, 'tell': 0.04,
            'like': 0.0, 'even': 0.04, 'spooky': 0.0, 'though': 0.04, 'psychiatrist': 0.04,
            'father': 0.04, 'stories': 0.08, 'ugly': 0.0, 'loved': 0.0, 'lived': 0.0,
            'thought': 0.0, 'ghost': 0.0, 'used': 0.04, 'kid': 0.0, 'ghosts': 0.04,
            'lots': 0.0, 'know': 0.04, 'smart': 0.0, 'library': 0.0, 'make': 0.0, 'cat': 0.0
        }

        document_tokens = [
            'father', 'used', 'tell', 'stories', 'sister', 'unusual',
            'talent', 'could', 'speak', 'ghosts', 'cats', 'know',
            'unusual', 'one', 'day', 'told', 'psychiatrist', 'okay',
            'still', 'love', 'cats', 'stories', 'even', 'though', 'scared'
        ]

        actual = calculate_tf(vocabulary, document_tokens)
        for token, frequency in expected.items():
            self.assertAlmostEqual(frequency, actual[token], delta=1e-4)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_bad_input(self):
        """
        Bad input scenario
        """
        expected = None
        bad_inputs = [None, True, 42, 3.14, (), 'document', {}, [],
                      [False, 5, {}, 'no!'], [[1.42], ['maybe']],
                      [[['Three layers!']]]]
        good_input = ['one', 'good', 'document']

        for bad_input in bad_inputs:
            actual = calculate_tf(bad_input, good_input)
            self.assertEqual(expected, actual)

            actual = calculate_tf(good_input, bad_input)
            self.assertEqual(expected, actual)

            actual = calculate_tf(bad_input, bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_return_value(self):
        """
        Function return value check
        """
        actual = calculate_tf(['this', 'vocabulary', 'is', 'not', 'lonely'],
                              ['it', 'has', 'a', 'friend', 'document'])

        self.assertIsInstance(actual, dict)
        for token, frequency in actual.items():
            self.assertIsInstance(token, str)
            self.assertIsInstance(frequency, float)
