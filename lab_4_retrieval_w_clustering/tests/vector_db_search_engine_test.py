"""
Checks the fourth lab's VectorDBSearchEngine class.
"""

# pylint: disable=protected-access,duplicate-code
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest

from lab_4_retrieval_w_clustering.main import DocumentVectorDB, VectorDBSearchEngine


class VectorDBSearchEngineTest(unittest.TestCase):
    """
    Tests VectorDBSearchEngine class functionality.
    """

    def setUp(self) -> None:
        self.stop_words = ["булочек"]
        self._document_vector_db = DocumentVectorDB(self.stop_words)
        self.corpus = ["съешь ещё этих французских булочек", "да выпей ещё чаю", "чаю"]
        self._document_vector_db.put_corpus(self.corpus)
        self._vector_search_engine = VectorDBSearchEngine(self._document_vector_db)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_vector_search_engine_init(self):
        """
        Ideal VectorDBSearchEngine scenario init
        """
        self.assertIsInstance(self._vector_search_engine._db, DocumentVectorDB)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_ideal(self):
        """
        Ideal retrieve_relevant_documents scenario
        """
        query = "ещё чаю"
        n = 3

        expected = [
            (0.6146142752620154, "чаю"),
            (0.6661661953007654, "да выпей ещё чаю"),
            (0.9887748724546137, "съешь ещё этих французских булочек"),
        ]
        actual = self._vector_search_engine.retrieve_relevant_documents(query, n)

        for pair_index, expected_pair in enumerate(expected):
            expected_distance = expected_pair[0]
            expected_text = expected_pair[1]
            actual_distance = actual[pair_index][0]
            actual_text = actual[pair_index][1]

            np.testing.assert_almost_equal(expected_distance, actual_distance)
            self.assertEqual(expected_text, actual_text)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_bm25_calls_func_from_2_lab(self):
        """
        Checks that VectorDBSearchEngine calls calculate_knn from lab_3
        """
        return_value = [(2, 0.6146142752620154), (1, 0.6661661953007654), (0, 0.9887748724546137)]
        self._vector_search_engine._calculate_knn = MagicMock(return_value=return_value)
        query = "ещё чаю"
        n = 3

        self._vector_search_engine.retrieve_relevant_documents(query, n)

        self._vector_search_engine._calculate_knn.assert_called()
        self._vector_search_engine._calculate_knn.assert_called_once()

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_invalid_input(self):
        """
        Invalid input scenario in retrieve_relevant_documents
        """
        query_bad_input = []
        n_bad_inputs = [-1, 0, 0.5]
        self.assertRaises(
            ValueError, self._vector_search_engine.retrieve_relevant_documents, query_bad_input, 1
        )
        for bad_input in n_bad_inputs:
            self.assertRaises(
                ValueError,
                self._vector_search_engine.retrieve_relevant_documents,
                "test",
                bad_input,
            )

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_tokenize_returns_none(self):
        """
        Invalid input scenario in retrieve_relevant_documents
        """
        self._vector_search_engine._tokenizer.tokenize = MagicMock(return_value=None)
        self.assertRaises(
            ValueError, self._vector_search_engine.retrieve_relevant_documents, "test", 1
        )

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_vectorize_returns_none(self):
        """
        Invalid input scenario in retrieve_relevant_documents
        """
        self._vector_search_engine._vectorizer.vectorize = MagicMock(return_value=None)
        self.assertRaises(
            ValueError, self._vector_search_engine.retrieve_relevant_documents, "test", 1
        )

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_calculate_knn_returns_none(self):
        """
        Invalid input scenario in retrieve_relevant_documents
        """
        self._vector_search_engine._calculate_knn = MagicMock(return_value=None)
        self.assertRaises(
            ValueError, self._vector_search_engine.retrieve_relevant_documents, "test", 1
        )
