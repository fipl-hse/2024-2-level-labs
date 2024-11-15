# pylint: disable=duplicate-code, protected-access
"""
Checks the third lab's AdvancedSearchEngine class.
"""

import json
import unittest
from pathlib import Path

import pytest

from lab_3_ann_retriever.main import AdvancedSearchEngine, Tokenizer, Vectorizer
from lab_3_ann_retriever.tests.tokenizer_test import get_documents_assets


class AdvancedSearchEngineTest(unittest.TestCase):
    """
    Tests Advanced Search Engine class functionality.
    """

    def setUp(self) -> None:
        with open(
            Path(__file__).parent.parent / "assets" / "stopwords.txt", "r", encoding="utf-8"
        ) as stop_words:
            stopwords = [line.strip() for line in stop_words]
        self._tokenizer = Tokenizer(stopwords)
        self.documents = get_documents_assets()[0]
        self.tokenized_docs = self._tokenizer.tokenize_documents(self.documents)
        if self.tokenized_docs is None:
            raise ValueError("Tokens are empty")
        self._vectorizer = Vectorizer(self.tokenized_docs)
        self._vectorizer.build()
        self.retriever = AdvancedSearchEngine(self._vectorizer, self._tokenizer)
        self.retriever.index_documents(self.documents)
        self.expected_path = Path(__file__).parent.parent / "tests" / "assets" / "correct_tree.json"
        self.test_path = Path(__file__).parent.parent / "test_tmp" / "tree.json"
        self.test_path.parent.mkdir(parents=True, exist_ok=True)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_ideal(self) -> None:
        """
        Ideal scenario for saving tree to file
        """
        self.retriever.save(str(self.test_path))

        with open(str(self.test_path), "r", encoding="utf-8") as f:
            actual = json.load(f)

        with open(str(self.expected_path), "r", encoding="utf-8") as f:
            expected = json.load(f)
        self.assertEqual(actual, expected)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_invalid_input(self) -> None:
        """
        Invalid input scenario for saving tree to file
        """
        bad_inputs = [{}, [], (), 7, 0.7, False]
        for bad_input in bad_inputs:
            actual = self.retriever.save(bad_input)
            self.assertFalse(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_return_value(self) -> None:
        """
        Checks return type for save method
        """
        self.assertIsInstance(self.retriever.save(str(self.test_path)), bool)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_ideal(self) -> None:
        """
        Ideal scenario for loading file
        """
        new_retriever = AdvancedSearchEngine(self._vectorizer, self._tokenizer)
        new_retriever.load(str(self.expected_path))
        self.assertEqual(self.retriever._documents, new_retriever._documents)
        self.assertEqual(self.retriever._document_vectors, new_retriever._document_vectors)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_invalid_input(self) -> None:
        """
        Invalid input scenario for loading file
        """
        bad_inputs = [{}, [], (), 7, 0.7, False]
        for bad_input in bad_inputs:
            actual = self.retriever.load(bad_input)
            self.assertIs(actual, False)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_return_value(self) -> None:
        """
        Checks return type for load method
        """
        actual = self.retriever.load(str(self.expected_path))
        self.assertIsInstance(actual, bool)
        self.assertTrue(actual)
