# pylint: disable=duplicate-code
"""
Checks the second lab's BM25 calculation function
"""
import json
import unittest
from pathlib import Path

import pytest

from lab_2_retrieval_w_bm25.main import calculate_bm25


class CalculateBM25Test(unittest.TestCase):
    """
    Tests for BM25 calculation functions
    """

    def setUp(self) -> None:
        self.vocabulary = [
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
        ]

        self.tokenized_documents = [
            ["upon", "time", "tale", "boy", "smart", "kind"],
            ["stories", "need", "told"],
            ["story", "began", "coffee", "shop", "across", "street"],
            ["stories", "love", "reading", "writing"],
            ["friend", "opened", "library", "named", "stories"],
            ["function", "must", "output", "string", "particular", "format"],
            ["stories", "beautiful", "stories", "ugly"],
            ["thought", "saw", "ghost", "cat"],
            ["love", "cat", "lived", "library", "across", "street"],
            ["library", "cat", "loved", "stories", "beautiful", "cat"],
        ]

        with open(Path(__file__).parent / "assets" / "test_scores.json", encoding="utf-8") as file:
            scores = json.load(file)
        self.idf = scores["IDF"]

        self.avg_doc_len = 5.0
        self.doc_len = len(self.tokenized_documents[9])

        self.bm25_scores = scores["BM25"]

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_bm25_ideal(self):
        """
        Ideal scenario
        """
        expected = self.bm25_scores["Document 9"]

        actual = calculate_bm25(
            self.vocabulary,
            self.tokenized_documents[9],
            self.idf,
            k1=1.5,
            b=0.75,
            avg_doc_len=self.avg_doc_len,
            doc_len=self.doc_len,
        )

        for token, frequency in expected.items():
            self.assertAlmostEqual(frequency, actual[token], delta=1e-4)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_bm25_all_documents(self):
        """
        Scenario for all documents
        """
        for n in range(10):
            actual = calculate_bm25(
                self.vocabulary,
                self.tokenized_documents[n],
                self.idf,
                k1=1.5,
                b=0.75,
                avg_doc_len=self.avg_doc_len,
                doc_len=len(self.tokenized_documents[n]),
            )

            for token, frequency in self.bm25_scores[f"Document {n}"].items():
                self.assertAlmostEqual(frequency, actual[token], delta=1e-4)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_bm25_new_document(self):
        """
        Scenario for a new document
        """
        expected = {
            "need": 0.0,
            "friend": 0.0,
            "time": 0.0,
            "tale": 0.0,
            "new": 0.0,
            "named": 0.0,
            "story": 0.0,
            "opened": 0.0,
            "cat": 0.0,
            "lived": 0.0,
            "output": 0.0,
            "writing": 0.0,
            "across": 0.0,
            "library": 0.0,
            "loved": 0.0,
            "completely": 0.0,
            "stories": 0.0,
            "saw": 0.0,
            "upon": 0.0,
            "smart": 0.0,
            "kind": 0.0,
            "began": 0.0,
            "particular": 0.0,
            "love": 0.0,
            "format": 0.0,
            "told": 0.0,
            "must": 0.0,
            "reading": 0.0,
            "thought": 0.0,
            "beautiful": 0.0,
            "ugly": 0.0,
            "shop": 0.0,
            "document": 0.0,
            "coffee": 0.0,
            "ghost": 0.0,
            "string": 0.0,
            "street": 0.0,
            "boy": 0.0,
            "function": 0.0,
        }

        actual = calculate_bm25(
            self.vocabulary,
            ["completely", "new", "document"],
            self.idf,
            k1=1.5,
            b=0.75,
            avg_doc_len=self.avg_doc_len,
            doc_len=self.doc_len,
        )

        for token, score in expected.items():
            self.assertEqual(score, actual[token])

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_bm25_bad_input(self):
        """
        Bad input scenario
        """
        expected = None
        bad_float_inputs = [None, True, 42, "string", (), {}, []]
        bad_int_inputs = [None, True, 3.14, "string", (), {}, []]
        bad_list_inputs = [None, True, 42, 3.14, "string", (), {}, [], [1, 3, 5, "seven"], [[]]]
        bad_dict_inputs = [
            None,
            True,
            42,
            3.14,
            "string",
            (),
            {},
            [],
            {1: 3, 5: "seven"},
            {"token": {"not": "frequency"}},
            {"bad frequency type": 10},
        ]

        for bad_float_input in bad_float_inputs:
            self.assertEqual(
                expected,
                calculate_bm25(
                    self.vocabulary,
                    self.tokenized_documents[9],
                    self.idf,
                    k1=bad_float_input,
                    b=0.75,
                    avg_doc_len=self.avg_doc_len,
                    doc_len=self.doc_len,
                ),
            )

            self.assertEqual(
                expected,
                calculate_bm25(
                    self.vocabulary,
                    self.tokenized_documents[9],
                    self.idf,
                    k1=1.5,
                    b=bad_float_input,
                    avg_doc_len=self.avg_doc_len,
                    doc_len=self.doc_len,
                ),
            )

            self.assertEqual(
                expected,
                calculate_bm25(
                    self.vocabulary,
                    self.tokenized_documents[9],
                    self.idf,
                    k1=1.5,
                    b=0.75,
                    avg_doc_len=bad_float_input,
                    doc_len=self.doc_len,
                ),
            )

        for bad_int_input in bad_int_inputs:
            self.assertEqual(
                expected,
                calculate_bm25(
                    self.vocabulary,
                    self.tokenized_documents[9],
                    self.idf,
                    k1=1.5,
                    b=0.75,
                    avg_doc_len=self.avg_doc_len,
                    doc_len=bad_int_input,
                ),
            )

        for bad_list_input in bad_list_inputs:
            self.assertEqual(
                expected,
                calculate_bm25(
                    bad_list_input,
                    self.tokenized_documents[9],
                    self.idf,
                    k1=1.5,
                    b=0.75,
                    avg_doc_len=self.avg_doc_len,
                    doc_len=self.doc_len,
                ),
            )
            self.assertEqual(
                expected,
                calculate_bm25(
                    self.vocabulary,
                    bad_list_input,
                    self.idf,
                    k1=1.5,
                    b=0.75,
                    avg_doc_len=self.avg_doc_len,
                    doc_len=self.doc_len,
                ),
            )

        for bad_dict_input in bad_dict_inputs:
            self.assertEqual(
                expected,
                calculate_bm25(
                    self.vocabulary,
                    self.tokenized_documents[9],
                    bad_dict_input,
                    k1=1.5,
                    b=0.75,
                    avg_doc_len=self.avg_doc_len,
                    doc_len=self.doc_len,
                ),
            )

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_bm25_return_value(self):
        """
        Function return value check
        """
        actual = calculate_bm25(
            self.vocabulary,
            self.tokenized_documents[9],
            self.idf,
            k1=1.5,
            b=0.75,
            avg_doc_len=self.avg_doc_len,
            doc_len=self.doc_len,
        )
        self.assertIsInstance(actual, dict)
        for token, tf_idf in actual.items():
            self.assertIsInstance(token, str)
            self.assertIsInstance(tf_idf, float)
