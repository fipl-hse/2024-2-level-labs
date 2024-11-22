# pylint: disable=protected-access
"""
Checks the fourth lab's DocumentVectorDB class.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lab_3_ann_retriever.main import Tokenizer
from lab_4_retrieval_w_clustering.main import BM25Vectorizer, DocumentVectorDB


class DocumentVectorDBTest(unittest.TestCase):
    """
    Tests DocumentVectorDB class functionality.
    """

    def setUp(self) -> None:
        self.stop_words = ["булочки"]
        self.tokenized_corpus = [
            ["съешь", "ещё", "этих", "французских", "булочек"],
            ["да", "выпей", "ещё", "чаю"],
        ]

        self.corpus = [
            "съешь ещё этих французских булочек",
            "да выпей ещё чаю",
        ]
        self.tokens = [
            ["съешь", "ещё", "этих", "французских", "булочек"],
            ["да", "выпей", "ещё", "чаю"],
        ]
        self.vectors = [
            (0.0, 0.0, 0.0, -1.5327980118420002, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, -1.6941451709832633, 0.0, 0.0, 0.0, 0.0),
        ]
        self._document_vector_db = DocumentVectorDB(self.stop_words)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_document_vector_db_initial(self):
        """
        Initialization DocumentVectorDB scenario
        """
        self.assertEqual(self._document_vector_db._DocumentVectorDB__documents, [])
        self.assertEqual(self._document_vector_db._DocumentVectorDB__vectors, {})
        self.assertEqual(self._document_vector_db._tokenizer._stop_words, self.stop_words)

        self.assertIsInstance(self._document_vector_db._DocumentVectorDB__documents, list)
        self.assertIsInstance(self._document_vector_db._DocumentVectorDB__vectors, dict)
        self.assertIsInstance(self._document_vector_db._vectorizer, BM25Vectorizer)
        self.assertIsInstance(self._document_vector_db._tokenizer, Tokenizer)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_put_corpus_ideal(self):
        """
        Ideal put_corpus scenario
        """
        expected_vectors = {0: self.vectors[0], 1: self.vectors[1]}

        self._document_vector_db._tokenizer.tokenize = MagicMock(side_effect=self.tokens)
        self._document_vector_db._vectorizer.vectorize = MagicMock(side_effect=self.vectors)

        self._document_vector_db.put_corpus(self.corpus)

        self.assertEqual(self._document_vector_db._DocumentVectorDB__documents, self.corpus)
        self.assertEqual(self._document_vector_db._DocumentVectorDB__vectors, expected_vectors)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_put_corpus_empty_input(self):
        """
        Invalid input scenario in put_corpus
        """
        bad_input = []
        self.assertRaises(ValueError, self._document_vector_db.put_corpus, bad_input)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_put_corpus_tokenize_return_none(self):
        """
        Scenario with put_corpus when tokenize returns None
        """
        self._document_vector_db._tokenizer.tokenize = MagicMock(return_value=None)
        self.assertRaises(ValueError, self._document_vector_db.put_corpus, self.corpus)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @patch("lab_4_retrieval_w_clustering.main.BM25Vectorizer.set_tokenized_corpus")
    @patch("lab_4_retrieval_w_clustering.main.BM25Vectorizer.build")
    def test_calculate_bm25_calls_func_from_2_lab(self, mock_build, mock_set_tokenized_corpus):
        """
        Checks that BM25 calls calculate_bm25 from lab_2
        """
        self._document_vector_db._tokenizer.tokenize = MagicMock(side_effect=self.tokens)
        self._document_vector_db._vectorizer.vectorize = MagicMock(side_effect=self.vectors)

        self._document_vector_db.put_corpus(self.corpus)

        mock_set_tokenized_corpus.assert_called_once()
        mock_set_tokenized_corpus.assert_called_once_with(self.tokens)

        mock_build.assert_called_once()

        self._document_vector_db._vectorizer.vectorize.assert_called()
        self._document_vector_db._tokenizer.tokenize.assert_called()

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_tokenizer_ideal(self):
        """
        Ideal get_tokenizer scenario
        """
        self.assertIsInstance(self._document_vector_db.get_tokenizer(), Tokenizer)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_vectorizer_ideal(self):
        """
        Ideal get_vectorizer scenario
        """
        self.assertIsInstance(self._document_vector_db.get_vectorizer(), BM25Vectorizer)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_vectors_ideal(self):
        """
        Ideal get_vectors scenario
        """

        self._document_vector_db._tokenizer.tokenize = MagicMock(side_effect=self.tokens)
        self._document_vector_db._vectorizer.vectorize = MagicMock(side_effect=self.vectors)

        self._document_vector_db.put_corpus(self.corpus)

        actual_vectors = self._document_vector_db.get_vectors((0, 1))

        for index, actual_pair in enumerate(actual_vectors):
            expected_vector = self.vectors[index]
            actual_vector = actual_pair[1]

            np.testing.assert_almost_equal(actual_vector, expected_vector)

        actual_vectors = self._document_vector_db.get_vectors()

        for index, actual_pair in enumerate(actual_vectors):
            expected_vector = self.vectors[index]
            actual_vector = actual_pair[1]

            np.testing.assert_almost_equal(actual_vector, expected_vector)

        actual_pairs = self._document_vector_db.get_vectors((0,))
        actual_vector = actual_pairs[0][1]
        np.testing.assert_almost_equal(actual_vector, self.vectors[0])

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_raw_documents_ideal(self):
        """
        Ideal get_raw_documents scenario
        """

        self._document_vector_db._tokenizer.tokenize = MagicMock(side_effect=self.tokens)
        self._document_vector_db._vectorizer.vectorize = MagicMock(side_effect=self.vectors)

        self._document_vector_db.put_corpus(self.corpus)

        actual_docs = self._document_vector_db.get_raw_documents()
        expected_docs = self._document_vector_db._DocumentVectorDB__documents
        self.assertEqual(expected_docs, actual_docs)

        actual_docs = self._document_vector_db.get_raw_documents((0,))
        expected_docs = [
            self._document_vector_db._DocumentVectorDB__documents[0],
        ]
        self.assertEqual(expected_docs, actual_docs)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_raw_documents_with_repeated_indices(self):
        """
        Scenario get_raw_documents with repeated indices
        """

        self._document_vector_db._tokenizer.tokenize = MagicMock(side_effect=self.tokens)
        self._document_vector_db._vectorizer.vectorize = MagicMock(side_effect=self.vectors)

        self._document_vector_db.put_corpus(self.corpus)

        actual_docs = self._document_vector_db.get_raw_documents((0, 0, 0, 0))
        expected_docs = [
            self._document_vector_db._DocumentVectorDB__documents[0],
        ]
        self.assertEqual(expected_docs, actual_docs)
