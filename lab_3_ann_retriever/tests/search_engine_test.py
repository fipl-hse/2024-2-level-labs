# pylint: disable=duplicate-code, protected-access
"""
Checks the third lab's Basic Search Engine class.
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pytest

from lab_3_ann_retriever.main import NaiveKDTree, SearchEngine, Tokenizer, Vectorizer
from lab_3_ann_retriever.tests.tokenizer_test import get_documents_assets


class SearchEngineTest(unittest.TestCase):
    """
    Tests Search Engine class functionality.
    """

    def setUp(self) -> None:
        with open(
            Path(__file__).parent.parent / "assets" / "stopwords.txt", "r", encoding="utf-8"
        ) as stop_words:
            stopwords = [line.strip() for line in stop_words]
        self._tokenizer = Tokenizer(stopwords)
        self.documents = get_documents_assets()[0]

        self.tokenized_docs = self._tokenizer.tokenize_documents(self.documents)
        self._vectorizer = Vectorizer(self.tokenized_docs)
        self._vectorizer.build()
        self.retriever = SearchEngine(self._vectorizer, self._tokenizer)
        self.query = "Мои кот и собака не дружат!"
        self.tree = NaiveKDTree()
        self.exp_path = Path(__file__).parent.parent / "tests" / "assets" / "engine_state.json"
        self.test_path = Path(__file__).parent.parent / "test_tmp" / "engine.json"
        self.test_path.parent.mkdir(parents=True, exist_ok=True)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_index_documents_ideal(self):
        """
        Ideal scenario for index_documents
        """
        self.retriever.index_documents(self.documents)

        actual = self.retriever._documents
        self.assertEqual(self.documents, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_index_documents_invalid_input(self):
        """
        Invalid input for index_documents scenario
        """
        expected = False
        bad_inputs = [[], {}, (), None, 7, 0.7, "", [7], [0.7]]
        for bad_input in bad_inputs:
            actual = self.retriever.index_documents(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_index_documents_return_value(self):
        """
        Checks return type for index_documents scenario
        """
        actual = self.retriever.index_documents(self.documents)
        self.assertIsInstance(actual, bool)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_ideal(self):
        """
        Ideal scenario for retrieving relevant documents
        """
        expected = [
            (
                0.4117127553769278,
                "Вектора используются для поиска релевантного документа. Давайте научимся, "
                "как их создавать и использовать!",
            )
        ]
        self.retriever.index_documents(self.documents)
        actual = self.retriever.retrieve_relevant_documents(self.query)
        for index, item in enumerate(expected):
            np.testing.assert_almost_equal(item[0], actual[index][0])
            self.assertEqual(actual[index][1], item[1])

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_invalid_input(self):
        """
        Invalid input scenario for retrieving relevant documents
        """
        expected = None
        bad_inputs = [[], {}, (), None, 0.7, "", 7, False]
        good_neighbours = 1
        for bad_input in bad_inputs:
            actual = self.retriever.retrieve_relevant_documents(
                query=bad_input, n_neighbours=good_neighbours
            )
            self.assertEqual(expected, actual)

            actual = self.retriever.retrieve_relevant_documents(query=self.query, n_neighbours=0)
            self.assertEqual(expected, actual)

            actual = self.retriever.retrieve_relevant_documents(query=self.query, n_neighbours=2)
            self.assertEqual(expected, actual)

            actual = self.retriever.retrieve_relevant_documents(query=bad_input, n_neighbours=-1)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_return_value(self):
        """
        Checks return type in retrieving relevant documents method
        """
        self.retriever.index_documents(self.documents)
        actual = self.retriever.retrieve_relevant_documents(self.query)

        self.assertIsInstance(actual, list)
        for doc in actual:
            self.assertIsInstance(doc, tuple)
            self.assertIsInstance(doc[0], float)
            self.assertIsInstance(doc[1], str)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_ideal(self):
        """
        Ideal scenario for save method
        """
        self.retriever.index_documents(self.documents)
        self.retriever.save(str(self.test_path))

        with open(str(self.test_path), "r", encoding="utf-8") as f:
            actual = json.load(f)

        with open(str(self.exp_path), "r", encoding="utf-8") as f:
            expected = json.load(f)
        self.assertEqual(actual, expected)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_invalid_input(self):
        """
        Invalid input scenario for save method
        """
        self.retriever.index_documents(self.documents)
        bad_inputs = [[], (), {}, 7, 0.7, False]
        for bad_input in bad_inputs:
            actual = self.retriever.save(bad_input)
            self.assertFalse(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_return_value(self):
        """
        Checks return type for save method
        """
        self.retriever.index_documents(self.documents)
        self.assertIsInstance(self.retriever.save(str(self.test_path)), bool)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_ideal(self) -> None:
        """
        Ideal scenario for loading file
        """
        self.retriever.index_documents(self.documents)
        new_retriever = SearchEngine(self._vectorizer, self._tokenizer)
        new_retriever.load(str(self.exp_path))
        self.assertEqual(self.retriever._documents, new_retriever._documents)
        self.assertEqual(self.retriever._document_vectors, new_retriever._document_vectors)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_missing_values(self) -> None:
        """
        Missing values scenario for loading file
        """
        self.retriever.index_documents(self.documents)
        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {
                "engine": {"tree": {}},
            }
            json.dump(state, f)

        loaded_retriever = SearchEngine(self._vectorizer, self._tokenizer)
        self.assertFalse(loaded_retriever.load(str(self.test_path)))

        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {
                "engine": {},
            }
            json.dump(state, f)

        loaded_retriever = SearchEngine(self._vectorizer, self._tokenizer)
        self.assertFalse(loaded_retriever.load(str(self.test_path)))

        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {
                "engine": {"documents": []},
            }
            json.dump(state, f)

        loaded_retriever = SearchEngine(self._vectorizer, self._tokenizer)
        self.assertFalse(loaded_retriever.load(str(self.test_path)))

        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {
                "engine": {"document_vectors": []},
            }
            json.dump(state, f)

        loaded_retriever = SearchEngine(self._vectorizer, self._tokenizer)
        self.assertFalse(loaded_retriever.load(str(self.test_path)))

        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {}
            json.dump(state, f)

        loaded_retriever = SearchEngine(self._vectorizer, self._tokenizer)
        self.assertFalse(loaded_retriever.load(str(self.test_path)))

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_return_value(self) -> None:
        """
        Checks return type for _load_documents method
        """
        self.retriever.index_documents(self.documents)
        actual = self.retriever.load(str(self.exp_path))
        self.assertIsInstance(actual, bool)
        self.assertTrue(actual)