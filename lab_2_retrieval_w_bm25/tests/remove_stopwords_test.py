"""
Checks the second lab's TF-IDF ranking function
"""
import unittest
from pathlib import Path

import pytest

from lab_2_retrieval_w_bm25.main import remove_stopwords, tokenize


class RemoveStopwordsTest(unittest.TestCase):
    """
    Tests for remove stopwords functions
    """
    def setUp(self) -> None:
        with open(Path(__file__).parent.parent / "assets" / "stopwords.txt",
                  "r", encoding="utf-8") as file:
            text = file.read()

        self.stopwords = text.split("\n")

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_remove_stopwords_ideal(self):
        """
        Ideal scenario
        """
        text = "My best friend once told me about a time he had met a ghost! " \
               "It might've been a lie, though. I wouldn't believe it myself! " \
               "They had just moved to the new house, and it was really old. " \
               "At night he was hearing laughter. It scared him, but his parents " \
               "didn't seem to hear anything. Time went by. But he never found " \
               "any resource that could laugh at night. So he decided to deal with it himself " \
               "his way. He started to talk to himself before bed. And soon enough, he heard " \
               "somebody answer him. This ghost called himself Arkady. " \
               "He said that he was just fond of scaring people, and my friend was" \
               " a good target. His parents thought he was just going insane. " \
               "And you wouldn't believe it yourself, until he would invite you over." \
               " Then you would hear two creatures laughing: Arkady and my friend." \
               " They just loved to scare people!"

        expected = [
            'best', 'friend', 'told', 'time', 'met', 'ghost', 'might', 've', 'lie',
            'though', 'wouldn', 't', 'believe', 'moved', 'new', 'house', 'really',
            'old', 'night', 'hearing', 'laughter', 'scared', 'parents', 'didn', 't',
            'seem', 'hear', 'anything', 'time', 'went', 'never', 'found', 'resource',
            'could', 'laugh', 'night', 'decided', 'deal', 'way', 'started', 'talk',
            'bed', 'soon', 'enough', 'heard', 'somebody', 'answer', 'ghost', 'called',
            'arkady', 'said', 'fond', 'scaring', 'people', 'friend', 'good', 'target',
            'parents', 'thought', 'going', 'insane', 'wouldn', 't', 'believe', 'would',
            'invite', 'would', 'hear', 'two', 'creatures', 'laughing', 'arkady', 'friend',
            'loved', 'scare', 'people'
        ]

        actual = remove_stopwords(tokenize(text), self.stopwords)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_remove_stopwords_bad_input(self):
        """
        Bad input scenario
        """
        expected = None
        bad_inputs = [None, True, 42, 3.14, (), {}, [], [1.7, 'No!']]

        for bad_input in bad_inputs:
            self.assertEqual(expected, remove_stopwords(bad_input, self.stopwords))
            self.assertEqual(expected, remove_stopwords(['normal', 'document'], bad_input))
        # print(remove_stopwords(bad_input, self.stopwords))

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_remove_stopwords_return_type(self):
        """
        Check return type
        """
        actual = remove_stopwords(['this', 'text', 'may', 'contain', 'stopwords'], self.stopwords)
        self.assertIsInstance(actual, list)
        for word in actual:
            self.assertIsInstance(word, str)
