# pylint: disable=duplicate-code
"""
Checks the second lab's calculate IDF function
"""

import unittest
from pathlib import Path

import pytest

from lab_2_retrieval_w_bm25.main import build_vocabulary, calculate_idf, remove_stopwords, tokenize


class CalculateIdfTest(unittest.TestCase):
    """
    Tests IDF calculation functions
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

        self.vocabulary = build_vocabulary(self.tokenized_documents)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_idf_ideal(self):
        """
        Ideal scenario
        """
        expected = {
            "upon": 1.8458,
            "time": 1.8458,
            "tale": 1.8458,
            "boy": 1.8458,
            "smart": 1.8458,
            "kind": 1.8458,
            "stories": 0.0,
            "need": 1.8458,
            "told": 1.8458,
            "story": 1.8458,
            "began": 1.8458,
            "coffee": 1.8458,
            "shop": 1.8458,
            "across": 1.2238,
            "street": 1.2238,
            "love": 1.2238,
            "reading": 1.8458,
            "writing": 1.8458,
            "friend": 1.8458,
            "opened": 1.8458,
            "library": 0.7621,
            "named": 1.8458,
            "function": 1.8458,
            "must": 1.8458,
            "output": 1.8458,
            "string": 1.8458,
            "particular": 1.8458,
            "format": 1.8458,
            "beautiful": 1.2238,
            "ugly": 1.8458,
            "thought": 1.8458,
            "saw": 1.8458,
            "ghost": 1.8458,
            "cat": 0.7621,
            "lived": 1.8458,
            "loved": 1.8458,
        }

        actual = calculate_idf(self.vocabulary, self.tokenized_documents)
        for token, frequency in actual.items():
            self.assertAlmostEqual(expected[token], frequency, delta=1e-4)

    def test_calculate_idf_new_tokens(self):
        """
        Scenario with adding new document with new tokens to the batch
        """
        expected = {
            "upon": 1.9459,
            "time": 1.9459,
            "tale": 1.9459,
            "boy": 1.9459,
            "smart": 1.9459,
            "kind": 1.9459,
            "stories": 0.1671,
            "need": 1.9459,
            "told": 1.9459,
            "story": 1.9459,
            "began": 1.9459,
            "coffee": 1.9459,
            "shop": 1.9459,
            "across": 1.335,
            "street": 1.335,
            "love": 1.335,
            "reading": 1.9459,
            "writing": 1.9459,
            "friend": 1.9459,
            "opened": 1.9459,
            "library": 0.8873,
            "named": 1.9459,
            "function": 1.9459,
            "must": 1.9459,
            "output": 1.9459,
            "string": 1.9459,
            "particular": 1.9459,
            "format": 1.9459,
            "beautiful": 1.335,
            "ugly": 1.9459,
            "thought": 1.9459,
            "saw": 1.9459,
            "ghost": 1.9459,
            "cat": 0.8873,
            "lived": 1.9459,
            "loved": 1.9459,
            "completely": 1.9459,
            "new": 1.9459,
            "document": 1.9459,
        }

        actual = calculate_idf(
            self.vocabulary, self.tokenized_documents + [["completely", "new", "document"]]
        )
        for token, frequency in actual.items():
            self.assertAlmostEqual(expected[token], frequency, delta=1e-4)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_idf_bad_input(self):
        """
        Bad input scenario
        """
        expected = None
        bad_vocabulary_inputs = [
            None,
            True,
            42,
            3.14,
            (),
            "document",
            {},
            [],
            [False, 5, {}, "no!"],
        ]
        bad_documents_inputs = [
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

        for bad_vocabulary_input in bad_vocabulary_inputs:
            actual = calculate_idf(bad_vocabulary_input, self.tokenized_documents)
            self.assertEqual(expected, actual)

        for bad_documents_input in bad_documents_inputs:
            actual = calculate_idf(self.vocabulary, bad_documents_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_idf_return_value(self):
        """
        Function return value check
        """
        actual = calculate_idf(self.vocabulary, self.tokenized_documents)

        self.assertIsInstance(actual, dict)
        for token, frequency in actual.items():
            self.assertIsInstance(token, str)
            self.assertIsInstance(frequency, float)
