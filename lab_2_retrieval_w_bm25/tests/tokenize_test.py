"""
Checks the second lab's text preprocessing functions
"""

import unittest

import pytest

from lab_2_retrieval_w_bm25.main import tokenize


class TokenizeTest(unittest.TestCase):
    """
    Tests tokenize function.
    """

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_ideal(self):
        """
        Ideal tokenize scenario
        """
        expected = ["the", "weather", "is", "sunny"]
        actual = tokenize("The weather is sunny.")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_several_sentences(self):
        """
        Tokenize text with several sentences
        """
        expected = [
            "the",
            "first",
            "sentence",
            "the",
            "second",
            "sentence",
            "the",
            "third",
            "sentence",
        ]
        actual = tokenize("The first sentence. The second sentence. The third sentence.")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_punctuation_marks(self):
        """
        Tokenize text with different punctuation marks
        """
        expected = ["the", "first", "sentence", "nice"]
        actual = tokenize("The, first sentence - nice!")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_russian_ideal(self):
        """
        Tokenize russian text
        """
        expected = ["этот", "текст", "на", "русском", "языке", "он", "содержит", "кириллицу"]
        actual = tokenize("Этот текст на русском языке. Он содержит кириллицу.")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_dirty_text(self):
        """
        Tokenize dirty text
        """
        expected = ["the", "first", "sentence", "the", "sec", "ond", "sent", "ence"]
        actual = tokenize("The first% sentence><. The sec&*ond sent@ence #.")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_text_without_words(self):
        """
        Text without words scenario
        """
        expected = []
        actual = tokenize("#%$ (2+3)/2!=2.6")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_bad_input(self):
        """
        Tokenize bad input argument scenario
        """
        bad_inputs = [[], {}, (), None, 7, 0.7, True]
        expected = None
        for bad_input in bad_inputs:
            actual = tokenize(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_return_value(self):
        """
        Tokenize return value check
        """
        actual = tokenize("This is a sentence, it is complex.")
        self.assertIsInstance(actual, list)
        self.assertTrue(all(isinstance(token, str) for token in actual))
