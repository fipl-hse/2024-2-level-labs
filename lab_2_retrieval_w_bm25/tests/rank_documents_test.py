# pylint: disable=duplicate-code
"""
Checks the second lab's ranking function
"""

import json
import unittest
from pathlib import Path
from unittest import mock

import pytest

from lab_2_retrieval_w_bm25.main import rank_documents


class RankDocumentsTest(unittest.TestCase):
    """
    Tests for documents ranking functions
    """

    def setUp(self) -> None:
        with open(Path(__file__).parent / "assets" / "test_scores.json", encoding="utf-8") as file:
            scores = json.load(file)

        self.tf_idf_scores = list(scores["TF-IDF"].values())
        self.bm_25_scores = list(scores["BM25"].values())

        self.query = "Best cat stories in the world!"

        with open(
            Path(__file__).parent.parent / "assets" / "stopwords.txt", "r", encoding="utf-8"
        ) as file:
            text = file.read()

        self.stopwords = text.split("\n")

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_rank_documents_ideal_bm25(self):
        """
        Ideal scenario for BM25
        """
        expected = [
            (9, 1.023),
            (7, 0.8375),
            (8, 0.6992),
            (0, 0.0),
            (1, 0.0),
            (2, 0.0),
            (3, 0.0),
            (4, 0.0),
            (5, 0.0),
            (6, 0.0),
        ]

        actual = rank_documents(self.bm_25_scores, self.query, self.stopwords)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_rank_documents_ideal_tf_idf(self):
        """
        Ideal scenario for TF-IDF
        """
        expected = [
            (9, 0.254),
            (7, 0.1905),
            (8, 0.127),
            (0, 0.0),
            (1, 0.0),
            (2, 0.0),
            (3, 0.0),
            (4, 0.0),
            (5, 0.0),
            (6, 0.0),
        ]

        actual = rank_documents(self.tf_idf_scores, self.query, self.stopwords)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_rank_documents_no_query_in_documents(self):
        """
        No query in documents scenario
        """
        expected = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]

        actual_tf_idf = rank_documents(self.tf_idf_scores, "big giraffes", self.stopwords)
        self.assertEqual(expected, actual_tf_idf)

        actual_bm25 = rank_documents(self.bm_25_scores, "big giraffes", self.stopwords)
        self.assertEqual(expected, actual_bm25)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_rank_documents_empty_tokenize(self):
        """
        Empty tokenize function scenario
        """
        expected = None
        with mock.patch("lab_2_retrieval_w_bm25.main.tokenize", return_value=None):
            actual = rank_documents(self.tf_idf_scores, "no query", self.stopwords)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_rank_documents_empty_remove_stopwords(self):
        """
        Empty remove stopwords function scenario
        """
        expected = None
        with mock.patch("lab_2_retrieval_w_bm25.main.remove_stopwords", return_value=None):
            actual = rank_documents(self.tf_idf_scores, "no query", self.stopwords)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_rank_documents_bad_input(self):
        """
        Bad input scenario
        """
        expected = None
        bad_document_list_inputs = [
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
        bad_query_inputs = [None, False, 24, 4.13, (), {}, [], ""]

        for bad_document_list_input in bad_document_list_inputs:
            actual = rank_documents(bad_document_list_input, self.query, self.stopwords)
            self.assertEqual(expected, actual)

        for bad_query_input in bad_query_inputs:
            actual = rank_documents(self.bm_25_scores, bad_query_input, self.stopwords)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_rank_documents_return_value(self):
        """
        Function return value check
        """
        actual = rank_documents(self.tf_idf_scores, "simple query", self.stopwords)
        self.assertIsInstance(actual, list)

        for document_index in actual:
            self.assertIsInstance(document_index, tuple)
            self.assertIsInstance(document_index[0], int)
            self.assertIsInstance(document_index[1], (int, float))
