"""
Checks the first lab text preprocessing functions
"""

import unittest

import pytest

from lab_1_classify_by_unigrams.main import tokenize


class TokenizeTest(unittest.TestCase):
    """
    Tests tokenize function
    """

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_ideal(self):
        """
        Ideal tokenize scenario
        """
        expected = ['t', 'h', 'e', 'w', 'e', 'a', 't', 'h', 'e', 'r', 'i', 's',
                    's', 'u', 'n', 'n', 'y']
        actual = tokenize('The weather is sunny.')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_several_sentences(self):
        """
        Tokenize text with several sentences
        """
        expected = ['t', 'h', 'e', 'f', 'i', 'r', 's', 't',
                    's', 'e', 'n', 't', 'e', 'n', 'c', 'e',
                    't', 'h', 'e', 's', 'e', 'c', 'o', 'n',
                    'd', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e']
        actual = tokenize('The first sentence. The second sentence.')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_punctuation_marks(self):
        """
        Tokenize text with different punctuation marks
        """
        expected = ['t', 'h', 'e', 'f', 'i', 'r', 's', 't', 's', 'e',
                    'n', 't', 'e', 'n', 'c', 'e', 'n', 'i', 'c', 'e']
        actual = tokenize('The, first sentence - nice!')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_deutsch_ideal(self):
        """
        Tokenize deutsch text
        """
        expected = ['i', 'c', 'h', 'w', 'e', 'i', 'ß',
                    'n', 'i', 'c', 'h', 't', 'w', 'a',
                    's', 'i', 'c', 'h', 'm', 'a', 'c',
                    'h', 'e', 'n', 'm', 'ö', 'c', 'h',
                    't', 'e', 'v', 'i', 'e', 'l', 'l',
                    'e', 'i', 'c', 'h', 't', 'i', 'c',
                    'h', 'm', 'u', 's', 's', 'd', 'a',
                    's', 'ü', 'b', 'e', 'r', 'l', 'e',
                    'g', 'e', 'n']
        actual = tokenize('Ich weiß nicht was ich machen möchte. Vielleicht ich muss das überlegen')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_dirty_text(self):
        """
        Tokenize dirty text
        """
        expected = ['t', 'h', 'e', 'f', 'i', 'r', 's',
                    't', 's', 'e', 'n', 't', 'e', 'n',
                    'c', 'e', 't', 'h', 'e', 's', 'e',
                    'c', 'o', 'n', 'd', 's', 'e', 'n',
                    't', 'e', 'n', 'c', 'e']
        actual = tokenize('The first% sentence><. The sec&*ond sent@ence #.')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_bad_input(self):
        """
        Tokenize bad input argument scenario
        """
        bad_inputs = [[], {}, (), None, 9, 9.34, True]
        expected = None
        for bad_input in bad_inputs:
            actual = tokenize(bad_input)
            self.assertEqual(expected, actual)
