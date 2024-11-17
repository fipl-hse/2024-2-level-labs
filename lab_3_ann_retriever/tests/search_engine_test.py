# pylint: disable=duplicate-code, protected-access
"""
Checks the third lab's Basic Search Engine class.
"""

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
    @pytest.mark.skip(reason="rework")
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_ideal(self):
        """
        Ideal scenario for retrieving relevant documents
        """
        expected = [
            (
                0.3287146667160643,
                "Моя собака думает, что ее любимый плед — это кошка. Просто он очень "
                "пушистый и мягкий. Забавно наблюдать, как они спят вместе!",
            ),
            (
                0.3342396698952498,
                "Мой кот Вектор по утрам приносит мне тапочки, а по вечерам мы гуляем с ним "
                "на шлейке во дворе. Вектор забавный и храбрый. Он не боится собак!",
            ),
        ]
        self.retriever.index_documents(self.documents)
        actual = self.retriever.retrieve_relevant_documents(self.query, 2)
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
        good_neighbours = 2
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
    @pytest.mark.skip(reason="rework")
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_return_value(self):
        """
        Checks return type in retrieving relevant documents method
        """
        self.retriever.index_documents(self.documents)
        actual = self.retriever.retrieve_relevant_documents(self.query, 2)

        self.assertIsInstance(actual, list)
        for doc in actual:
            self.assertIsInstance(doc, tuple)
            self.assertIsInstance(doc[0], float)
            self.assertIsInstance(doc[1], str)
