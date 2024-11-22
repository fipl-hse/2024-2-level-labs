"""
Checks the fourth lab's VectorDBEngine class.
"""

import unittest
from unittest.mock import MagicMock

import pytest

from lab_4_retrieval_w_clustering.main import VectorDBEngine


class VectorDBEngineTest(unittest.TestCase):
    """
    Tests VectorDBEngine class functionality.
    """

    def setUp(self) -> None:
        self.mock_db = MagicMock()
        self.mock_db.get_raw_documents.return_value = ["Document 1", "Document 2"]

        self.mock_engine = MagicMock()
        self.mock_engine.retrieve_relevant_documents.return_value = [
            (0.1, "Document 1"),
            (0.2, "Document 2"),
        ]

        self.vector_db_engine = VectorDBEngine(self.mock_db, self.mock_engine)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_ideal(self):
        """
        Test retrieval of relevant documents.
        """
        query = "test query"
        n_neighbours = 2
        result = self.vector_db_engine.retrieve_relevant_documents(query, n_neighbours)

        expected_result = [(0.1, "Document 1"), (0.2, "Document 2")]
        self.assertEqual(result, expected_result)
        self.mock_engine.retrieve_relevant_documents.assert_called_once_with(
            query, n_neighbours=n_neighbours
        )

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_no_results(self):
        """
        Test retrieval when no relevant documents are found.
        """
        self.mock_engine.retrieve_relevant_documents.return_value = None

        query = "nonexistent query"
        n_neighbours = 2
        result = self.vector_db_engine.retrieve_relevant_documents(query, n_neighbours)

        self.assertIsNone(result)
        self.mock_engine.retrieve_relevant_documents.assert_called_once_with(
            query, n_neighbours=n_neighbours
        )
