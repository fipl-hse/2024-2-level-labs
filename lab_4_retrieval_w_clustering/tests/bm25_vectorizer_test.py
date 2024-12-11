# pylint: disable=protected-access
"""
Checks the fourth lab's Vectorizer class.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from lab_4_retrieval_w_clustering.main import BM25Vectorizer


class BM25VectorizerTest(unittest.TestCase):
    """
    Tests BM25Vectorizer class functionality.
    """

    def setUp(self) -> None:
        self.vectorizer = BM25Vectorizer()
        self.tokenized_corpus = [
            ["съешь", "ещё", "этих", "французских", "булочек"],
            ["да", "выпей", "ещё", "чаю"],
        ]

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_avg_doc_len_initial(self):
        """
        Ideal avg_doc_len scenario
        """
        self.assertEqual(-1, self.vectorizer._avg_doc_len)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_set_tokenized_corpus_ideal(self):
        """
        Ideal set_tokenized_corpus scenario
        """
        self.vectorizer.set_tokenized_corpus(self.tokenized_corpus)

        actual_corpus = self.vectorizer._corpus
        actual_doc_len = self.vectorizer._avg_doc_len

        self.assertEqual(self.tokenized_corpus, actual_corpus)
        self.assertEqual(4.5, actual_doc_len)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_set_tokenized_corpus_empty_input(self):
        """
        Invalid input scenario in vectorize
        """
        bad_input = []
        self.assertRaises(ValueError, self.vectorizer.set_tokenized_corpus, bad_input)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_set_tokenized_corpus_repetition(self):
        """
        Checks that set_tokenized_corpus forgot about previous parameters and set new corpus
        """
        test_corpus = [
            ["это", "тестовые", "токены"],
        ]
        corpora = [test_corpus, self.tokenized_corpus]
        avg_lengths = [3.0, 4.5]

        for index, corpus in enumerate(corpora):
            self.vectorizer.set_tokenized_corpus(corpus)

            actual_corpus = self.vectorizer._corpus
            actual_doc_len = self.vectorizer._avg_doc_len

            self.assertEqual(corpus, actual_corpus)
            self.assertEqual(avg_lengths[index], actual_doc_len)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_bm25_ideal(self):
        """
        Ideal calculate_bm25 scenario
        """
        self.vectorizer.set_tokenized_corpus(self.tokenized_corpus)
        self.vectorizer.build()
        document = ["булочка", "франция", "ещё"]

        actual = self.vectorizer._calculate_bm25(document)
        expected = [0.0, 0.0, 0.0, -1.8934564, 0.0, 0.0, 0.0, 0.0]

        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @patch("lab_4_retrieval_w_clustering.main.calculate_bm25")
    def test_calculate_bm25_calls_func_from_2_lab(self, mock_calculate_bm25):
        """
        Checks that BM25 calls calculate_bm25 from lab_2
        """
        self.vectorizer.set_tokenized_corpus(self.tokenized_corpus)
        self.vectorizer.build()
        document = ["булочка", "франция", "ещё"]
        self.vectorizer._calculate_bm25(document)

        self.assertTrue(mock_calculate_bm25.called)
        self.assertEqual(mock_calculate_bm25.call_count, 1)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_bm25_wo_build(self):
        """
        Scenario calculate_bm25 without build
        """
        document = ["булочка", "франция", "ещё"]

        self.assertEqual(self.vectorizer._calculate_bm25(document), ())

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @patch("lab_4_retrieval_w_clustering.main.BM25Vectorizer._calculate_bm25")
    def test_calculate_vectorize_calls_calculate_bm25(self, mock_calculate_bm25):
        """
        Checks that BM25 calls calculate_bm25 from lab_2
        """
        self.vectorizer.set_tokenized_corpus(self.tokenized_corpus)
        self.vectorizer.build()
        document = ["булочка", "франция", "ещё"]
        self.vectorizer.vectorize(document)

        self.assertTrue(mock_calculate_bm25.called)
        self.assertEqual(mock_calculate_bm25.call_count, 1)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_vectorize_empty_input(self):
        """
        Invalid input scenario in vectorize
        """
        bad_input = []
        self.assertRaises(ValueError, self.vectorizer.vectorize, bad_input)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @patch("lab_4_retrieval_w_clustering.main.BM25Vectorizer._calculate_bm25")
    def test_vectorize_calculate_bm25_return_none(self, mock_calculate_bm25):
        """
        Vectorize scenario when calculate_bm25 returns None
        """
        mock_calculate_bm25.return_value = None

        self.vectorizer.set_tokenized_corpus(self.tokenized_corpus)
        self.vectorizer.build()
        document = ["булочка", "франция", "ещё"]
        self.assertRaises(ValueError, self.vectorizer.vectorize, document)
