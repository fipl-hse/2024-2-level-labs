# pylint: disable=duplicate-code
"""
Checks the second lab's document quantity functions
"""

import unittest
from pathlib import Path

import pytest

from lab_2_retrieval_w_bm25.main import build_vocabulary, remove_stopwords, tokenize


class BuildVocabularyTest(unittest.TestCase):
    """
    Tests document quantity function
    """

    def setUp(self) -> None:
        self.documents = [
            "Once upon a time there was a tale about a boy. He was smart and kind.",
            "Some stories need to be told.",
            "This story began in a coffee shop across the street.",
            "Stories! I love reading and writing them!",
            'My friend opened a library named "Stories101"!',
            "The function must output a string in a particular format.",
            "Some stories are beautiful. Some stories are ugly.",
            "Once I thought I saw a ghost! But it was just my cat!",
            "I love the cat that lived in the library across the street.",
            "Library cat loved stories and other beautiful cat.",
        ]

        with open(
            Path(__file__).parent.parent / "assets/stopwords.txt", "r", encoding="utf-8"
        ) as file:
            stopwords = file.read().split("\n")

        self.tokenized_documents = [
            remove_stopwords(tokenize(doc), stopwords) for doc in self.documents
        ]

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_vocabulary_ideal(self):
        """
        Ideal scenario
        """
        expected = {
            "upon",
            "time",
            "tale",
            "boy",
            "smart",
            "kind",
            "stories",
            "need",
            "told",
            "story",
            "began",
            "coffee",
            "shop",
            "across",
            "street",
            "love",
            "reading",
            "writing",
            "friend",
            "opened",
            "library",
            "named",
            "function",
            "must",
            "output",
            "string",
            "particular",
            "format",
            "beautiful",
            "ugly",
            "thought",
            "saw",
            "ghost",
            "cat",
            "lived",
            "loved",
        }

        actual = build_vocabulary(self.tokenized_documents)
        self.assertEqual(expected, set(actual))

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_vocabulary_bad_input(self):
        """
        Bad input scenario
        """
        expected = None
        bad_inputs = [
            None,
            True,
            42,
            3.14,
            (),
            "document",
            {},
            [],
            [False, 5, {}, "no!"],
            [[1.42], ["maybe"]],
            [[["Three layers!"]]],
        ]

        for bad_input in bad_inputs:
            actual = build_vocabulary(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_vocabulary_return_value(self):
        """
        Function return value check
        """
        actual = build_vocabulary(self.tokenized_documents)
        self.assertIsInstance(actual, list)
        for token in actual:
            self.assertIsInstance(token, str)
