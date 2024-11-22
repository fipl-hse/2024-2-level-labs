"""
Checks the fourth lab's AdvancedSearchEngine class.
"""

# pylint: disable=protected-access
import unittest
from unittest.mock import MagicMock

import pytest

from lab_3_ann_retriever.main import SearchEngine
from lab_4_retrieval_w_clustering.main import VectorDBAdvancedSearchEngine


class VectorDBAdvancedSearchEngineTest(unittest.TestCase):
    """
    Tests VectorDBAdvancedSearchEngine class functionality.
    """

    def setUp(self) -> None:
        self.mock_db = MagicMock()
        self.mock_vectorizer = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_db.get_vectorizer.return_value = self.mock_vectorizer
        self.mock_db.get_tokenizer.return_value = self.mock_tokenizer

        self.advanced_engine = VectorDBAdvancedSearchEngine(self.mock_db)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_initialization(self):
        """
        Test initialization of VectorDBAdvancedSearchEngine instance.
        """
        self.assertIsInstance(self.advanced_engine._engine, SearchEngine)
