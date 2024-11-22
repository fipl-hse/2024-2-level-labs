"""
Checks the fourth lab's VectorDBTreeSearchEngine class.
"""

# pylint: disable=protected-access, duplicate-code
import unittest
from unittest.mock import MagicMock

import pytest

from lab_3_ann_retriever.main import SearchEngine
from lab_4_retrieval_w_clustering.main import VectorDBTreeSearchEngine


class VectorDBTreeSearchEngineTest(unittest.TestCase):
    """
    Tests VectorDBTreeSearchEngine class functionality.
    """

    def setUp(self) -> None:
        self.mock_db = MagicMock()
        self.mock_vectorizer = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_db.get_vectorizer.return_value = self.mock_vectorizer
        self.mock_db.get_tokenizer.return_value = self.mock_tokenizer

        self.tree_engine = VectorDBTreeSearchEngine(self.mock_db)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_initialization(self):
        """
        Test initialization of VectorDBTreeSearchEngine instance.
        """
        self.assertIsInstance(self.tree_engine._engine, SearchEngine)
