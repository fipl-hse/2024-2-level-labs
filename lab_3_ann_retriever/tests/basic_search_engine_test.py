# pylint: disable=duplicate-code, protected-access, too-many-lines
"""
Checks the third lab's Basic Search Engine class.
"""

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from lab_3_ann_retriever.main import BasicSearchEngine, Tokenizer, Vectorizer
from lab_3_ann_retriever.tests.tokenizer_test import get_documents_assets


class BasicSearchEngineTest(unittest.TestCase):
    """
    Tests Basic Search Engine class functionality.
    """

    def setUp(self) -> None:
        with open(
            Path(__file__).parent.parent / "assets" / "stopwords.txt", "r", encoding="utf-8"
        ) as stop_words:
            stopwords = [line.strip() for line in stop_words]
        self._tokenizer = Tokenizer(stopwords)
        self.documents = get_documents_assets()[0]
        self.documents_vector = [
            (
                0.0,
                0.0,
                0.094,
                0.0,
                0.0,
                0.0,
                0.094,
                0.0,
                0.0,
                0.094,
                0.0,
                0.0,
                0.0,
                0.0,
                0.094,
                0.094,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.094,
                0.0,
                0.0,
                0.0,
                0.094,
                0.0,
                0.0,
                0.0,
                0.0,
                0.094,
                0.0,
                0.0,
                0.094,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                0.061,
                0.121,
                0.0,
                0.061,
                0.0,
                0.061,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.061,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.061,
                0.0,
                0.0,
                0.0,
                0.0,
                0.061,
                0.0,
                0.0,
                0.0,
                0.061,
                0.0,
                0.061,
                0.061,
                0.0,
            ),
            (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.077,
                0.077,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                0.0,
                0.0,
                0.0,
                0.0,
                0.061,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.061,
                0.061,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.061,
                0.061,
                0.061,
                0.061,
                0.0,
                0.0,
                0.0,
                0.061,
                0.0,
                0.0,
                0.061,
                0.0,
                0.0,
                0.0,
                0.0,
                0.061,
                0.0,
                0.061,
                0.0,
                0.0,
                0.0,
                0.0,
                0.061,
            ),
        ]

        self.tokenized_docs = self._tokenizer.tokenize_documents(self.documents)
        self._vectorizer = Vectorizer(self.tokenized_docs)
        self._vectorizer.build()
        self.retriever = BasicSearchEngine(self._vectorizer, self._tokenizer)
        self.query = "Мои кот и собака не дружат!"

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_index_document_ideal(self):
        """
        Ideal scenario for _index_document
        """
        expected = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.06052127574194312,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.06052127574194312,
            0.06052127574194312,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.06052127574194312,
            0.06052127574194312,
            0.06052127574194312,
            0.06052127574194312,
            0.0,
            0.0,
            0.0,
            0.06052127574194312,
            0.0,
            0.0,
            0.06052127574194312,
            0.0,
            0.0,
            0.0,
            0.0,
            0.06052127574194312,
            0.0,
            0.06052127574194312,
            0.0,
            0.0,
            0.0,
            0.0,
            0.06052127574194312,
        ]
        document = (
            "Моя собака думает, что ее любимый плед - это кошка. "
            "Просто он очень пушистый и мягкий. "
            "Забавно наблюдать, как они спят вместе! "
        )
        self.documents = get_documents_assets()[0]
        actual = self.retriever._index_document(document)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_index_document_invalid_input(self):
        """
        Invalid input for _index_document scenario
        """
        bad_inputs = [[], {}, (), None, 7, 0.7, "", False]
        for bad_input in bad_inputs:
            actual = self.retriever._index_document(bad_input)
            self.assertIsNone(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_index_document_return_value(self):
        """
        Checks return type for _index_document scenario
        """
        document = (
            "Моя собака думает, что ее любимый плед - это кошка. "
            "Просто он очень пушистый и мягкий. "
            "Забавно наблюдать, как они спят вместе! "
        )
        actual = self.retriever._index_document(document)
        self.assertIsInstance(actual, tuple)
        for num in actual:
            self.assertIsInstance(num, float)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
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
    @pytest.mark.mark6
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
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_index_documents_return_value(self):
        """
        Checks return type for index_documents scenario
        """
        actual = self.retriever.index_documents(self.documents)
        self.assertIsInstance(actual, bool)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_knn_ideal(self):
        """
        Ideal calculate_knn scenario
        """
        expected = [
            (0, 0.0),
            (2, 0.28382212739672014),
            (3, 0.28819264390334465),
            (1, 0.29341779087165115),
        ]
        query_vector = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.077,
            0.0,
            0.0,
            0.0,
            0.077,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.077,
            0.077,
            0.0,
            0.077,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.077,
            0.0,
            0.0,
            0.0,
            0.0,
            0.077,
            0.077,
            0.0,
            0.077,
            0.0,
            0.0,
            0.0,
        ]
        documents_vector = [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.077,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.077,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.049,
                0.049,
                0.049,
                0.0,
                0.021,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.021,
                0.0,
                0.049,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.099,
                0.049,
                0.021,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
            ],
            [
                0.063,
                0.0,
                0.0,
                0.0,
                0.0,
                0.026,
                0.0,
                0.0,
                0.063,
                0.0,
                0.0,
                0.0,
                0.063,
                0.026,
                0.0,
                0.0,
                0.063,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.026,
                0.0,
                0.0,
                0.026,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.063,
                0.0,
                0.0,
                0.0,
                0.0,
                0.026,
                0.0,
                0.0,
                0.063,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.049,
                0.0,
                0.049,
                0.0,
                0.049,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.0,
                0.0,
                0.049,
                0.0,
                0.049,
                0.021,
                0.0,
                0.0,
                0.0,
                0.049,
                0.049,
                0.0,
                0.049,
                0.0,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.0,
                0.021,
                0.0,
                0.0,
                0.0,
                0.049,
            ],
        ]
        actual = self.retriever._calculate_knn(query_vector, documents_vector, 4)
        self.assertEqual(expected[0][0], actual[0][0])
        np.testing.assert_almost_equal(actual[0][1], expected[0][1])

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_knn_invalid_input(self):
        """
        Invalid input scenario for calculate_knn
        """
        expected = None
        bad_inputs = [None]
        good_input_query = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.077, 0.0, 0.0, 0.0, 0.077]
        good_input_docs = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.077, 0.0, 0.0, 0.0, 0.077],
            [0.0, 0.049, 0.049, 0.049, 0.0, 0.021, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.063, 0.0, 0.0, 0.0, 0.0, 0.026, 0.0, 0.0, 0.063, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.049, 0.0, 0.0, 0.049, 0.0, 0.049, 0.0],
        ]
        good_input_neigbour = 4
        for bad_input in bad_inputs:
            actual = self.retriever._calculate_knn(
                query_vector=bad_input,
                document_vectors=good_input_docs,
                n_neighbours=good_input_neigbour,
            )
            self.assertEqual(expected, actual)

            actual = self.retriever._calculate_knn(
                query_vector=good_input_query,
                document_vectors=bad_input,
                n_neighbours=good_input_neigbour,
            )
            self.assertEqual(expected, actual)

            actual = self.retriever._calculate_knn(
                query_vector=bad_input, document_vectors=bad_input, n_neighbours=good_input_neigbour
            )
            self.assertEqual(expected, actual)

            actual = self.retriever._calculate_knn(
                query_vector=bad_input, document_vectors=good_input_docs, n_neighbours=0
            )
            self.assertEqual(expected, actual)

            actual = self.retriever._calculate_knn(
                query_vector=good_input_query, document_vectors=bad_input, n_neighbours=-1
            )
            self.assertEqual(expected, actual)

            actual = self.retriever._calculate_knn(
                query_vector=bad_input, document_vectors=bad_input, n_neighbours=0
            )
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_knn_return_value(self):
        """
        Checks return type for calculate_knn scenario
        """
        query_vector = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.077,
            0.0,
            0.0,
            0.0,
            0.077,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.077,
            0.077,
            0.0,
            0.077,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.077,
            0.0,
            0.0,
            0.0,
            0.0,
            0.077,
            0.077,
            0.0,
            0.077,
            0.0,
            0.0,
            0.0,
        ]
        documents_vector = [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.077,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
                0.0,
                0.077,
                0.077,
                0.0,
                0.077,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.049,
                0.049,
                0.049,
                0.0,
                0.021,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.021,
                0.0,
                0.049,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.099,
                0.049,
                0.021,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
            ],
            [
                0.063,
                0.0,
                0.0,
                0.0,
                0.0,
                0.026,
                0.0,
                0.0,
                0.063,
                0.0,
                0.0,
                0.0,
                0.063,
                0.026,
                0.0,
                0.0,
                0.063,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.026,
                0.0,
                0.0,
                0.026,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.063,
                0.0,
                0.0,
                0.0,
                0.0,
                0.026,
                0.0,
                0.0,
                0.063,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.049,
                0.0,
                0.049,
                0.0,
                0.049,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.0,
                0.0,
                0.049,
                0.0,
                0.049,
                0.021,
                0.0,
                0.0,
                0.0,
                0.049,
                0.049,
                0.0,
                0.049,
                0.0,
                0.0,
                0.0,
                0.049,
                0.0,
                0.0,
                0.0,
                0.021,
                0.0,
                0.0,
                0.0,
                0.049,
            ],
        ]
        actual = self.retriever._calculate_knn(query_vector, documents_vector, 4)
        self.assertIsInstance(actual, list)
        for neighbour in actual:
            self.assertIsInstance(neighbour, tuple)
            self.assertIsInstance(neighbour[0], int)
            self.assertIsInstance(neighbour[1], float)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
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
        for index, expectation in enumerate(expected):
            self.assertEqual(expectation[1], actual[index][1])
            np.testing.assert_almost_equal(actual[index][0], expectation[0])

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
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

            actual = self.retriever.retrieve_relevant_documents(query=bad_input, n_neighbours=-1)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
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

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_none_calculate_knn(self):
        """
        Checks return of inner Basic Engine Search functions
        """
        with mock.patch.object(self.retriever, "_calculate_knn", return_value=None):
            actual = self.retriever.retrieve_relevant_documents(self.query, 2)
            self.assertIsNone(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_index_document_none_tokenizer(self):
        """
        Checks return of inner Basic Engine Search functions
        """
        document = (
            "Моя собака думает, что ее любимый плед - это кошка. "
            "Просто он очень пушистый и мягкий. "
            "Забавно наблюдать, как они спят вместе! "
        )
        with mock.patch.object(self.retriever._tokenizer, "tokenize", return_value=None):
            actual = self.retriever._index_document(document)
            self.assertIsNone(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_index_document_none_vectorizer(self):
        """
        Checks return of inner Basic Engine Search functions
        """
        document = (
            "Моя собака думает, что ее любимый плед - это кошка. "
            "Просто он очень пушистый и мягкий. "
            "Забавно наблюдать, как они спят вместе! "
        )
        with mock.patch.object(self.retriever._vectorizer, "vectorize", return_value=None):
            actual = self.retriever._index_document(document)
            self.assertIsNone(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_relevant_documents_none_vectorizer(self):
        """
        Checks return of inner Basic Engine Search functions
        """
        with mock.patch.object(self.retriever._vectorizer, "vectorize", return_value=None):
            actual = self.retriever.retrieve_relevant_documents(self.query, 2)
            self.assertIsNone(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_empty_knn_results(self):
        """
        Tests the case where _calculate_knn returns an empty list.
        """
        with mock.patch.object(self.retriever, "_calculate_knn", return_value=[]):
            result = self.retriever.retrieve_relevant_documents(self.query, 2)
            self.assertIsNone(result)
