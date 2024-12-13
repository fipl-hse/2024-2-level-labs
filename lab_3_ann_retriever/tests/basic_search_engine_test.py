# pylint: disable=duplicate-code, protected-access, too-many-lines, too-many-public-methods
"""
Checks the third lab's Basic Search Engine class.
"""
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from lab_3_ann_retriever___vhkefbckemvkjo.main import BasicSearchEngine, Tokenizer, Vectorizer
from lab_3_ann_retriever___vhkefbckemvkjo.tests.tokenizer_test import get_documents_assets


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
        self.dump = {
            "document_vectors": [
                {
                    "elements": {
                        2: 0.09414420670968929,
                        6: 0.09414420670968929,
                        9: 0.09414420670968929,
                        14: 0.09414420670968929,
                        15: 0.09414420670968929,
                        23: 0.09414420670968929,
                        27: 0.09414420670968929,
                        32: 0.09414420670968929,
                        35: 0.09414420670968929,
                    },
                    "len": 42,
                },
                {
                    "elements": {
                        0: 0.06052127574194312,
                        1: 0.12104255148388623,
                        3: 0.06052127574194312,
                        5: 0.06052127574194312,
                        16: 0.06052127574194312,
                        28: 0.06052127574194312,
                        33: 0.06052127574194312,
                        37: 0.06052127574194312,
                        39: 0.06052127574194312,
                        40: 0.06052127574194312,
                    },
                    "len": 42,
                },
                {
                    "elements": {
                        8: 0.07702707821701851,
                        13: 0.07702707821701851,
                        17: 0.07702707821701851,
                        18: 0.07702707821701851,
                        24: 0.07702707821701851,
                        31: 0.07702707821701851,
                    },
                    "len": 42,
                },
                {
                    "elements": {
                        4: 0.06052127574194312,
                        10: 0.06052127574194312,
                        11: 0.06052127574194312,
                        19: 0.06052127574194312,
                        20: 0.06052127574194312,
                        21: 0.06052127574194312,
                        22: 0.06052127574194312,
                        26: 0.06052127574194312,
                        29: 0.06052127574194312,
                        34: 0.06052127574194312,
                        36: 0.06052127574194312,
                        41: 0.06052127574194312,
                    },
                    "len": 42,
                },
            ],
            "documents": [
                "Вектора используются для поиска релевантного документа. "
                "Давайте научимся, как их создавать и использовать!",
                "Мой кот Вектор по утрам приносит мне тапочки, а по вечерам мы "
                "гуляем с ним на шлейке во дворе. Вектор забавный и храбрый. Он "
                "не боится собак!",
                "Котёнок, которого мы нашли во дворе, очень забавный и "
                "пушистый. По утрам я играю с ним в догонялки перед работой.",
                "Моя собака думает, что ее любимый плед — это кошка. Просто он "
                "очень пушистый и мягкий. Забавно наблюдать, как они спят "
                "вместе!",
            ],
        }
        self.exp_path = Path(__file__).parent.parent / "tests" / "assets" / "basic_data.json"
        self.test_path = Path(__file__).parent.parent / "test_tmp" / "basic_state.json"
        self.test_path.parent.mkdir(parents=True, exist_ok=True)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_index_document_ideal(self):
        """
        Ideal scenario for _index_document
        """
        expected = (
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
        )
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
        Ideal _calculate_knn scenario
        """
        expected = [
            (0, 0.0),
            (2, 0.28382212739672014),
            (3, 0.28819264390334465),
            (1, 0.29341779087165115),
        ]
        query_vector = (
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
        )
        documents_vector = [
            (
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
            ),
            (
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
            ),
            (
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
            ),
            (
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
            ),
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
        Invalid input scenario for _calculate_knn
        """
        expected = None
        bad_inputs = [None]
        good_input_query = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.077, 0.0, 0.0, 0.0, 0.077)
        good_input_docs = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.077, 0.0, 0.0, 0.0, 0.077),
            (0.0, 0.049, 0.049, 0.049, 0.0, 0.021, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.063, 0.0, 0.0, 0.0, 0.0, 0.026, 0.0, 0.0, 0.063, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.049, 0.0, 0.0, 0.049, 0.0, 0.049, 0.0),
        ]
        good_input_neighbour = 4
        for bad_input in bad_inputs:
            actual = self.retriever._calculate_knn(
                query_vector=bad_input,
                document_vectors=good_input_docs,
                n_neighbours=good_input_neighbour,
            )
            self.assertEqual(expected, actual)

            actual = self.retriever._calculate_knn(
                query_vector=good_input_query,
                document_vectors=bad_input,
                n_neighbours=good_input_neighbour,
            )
            self.assertEqual(expected, actual)

            actual = self.retriever._calculate_knn(
                query_vector=bad_input,
                document_vectors=bad_input,
                n_neighbours=good_input_neighbour,
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
        Checks return type for _calculate_knn scenario
        """
        query_vector = (
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
        )
        documents_vector = [
            (
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
            ),
            (
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
            ),
            (
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
            ),
            (
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
            ),
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
    def test_empty_distances(self):
        """
        Handles the case when distance is empty
        """
        query_vector = (
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
        )
        document_vectors = []
        n_neighbors = 2

        with mock.patch("lab_3_ann_retriever___vhkefbckemvkjo.main.calculate_distance", return_value=None):
            result = self.retriever._calculate_knn(query_vector, document_vectors, n_neighbors)
        self.assertIsNone(result)

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
        None _calculate_knn method return scenario
        """
        with mock.patch.object(self.retriever, "_calculate_knn", return_value=None):
            actual = self.retriever.retrieve_relevant_documents(self.query, 2)
            self.assertIsNone(actual)

        with mock.patch.object(self.retriever, "_calculate_knn", return_value=[(None, None)]):
            actual = self.retriever.retrieve_relevant_documents(self.query, 2)
            self.assertIsNone(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_index_document_none_tokenizer(self):
        """
        None tokenize method return scenario
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
        None vectorize method return scenario
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
        None vectorize method return scenario
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
        Tests the case where _calculate_knn returns an empty list
        """
        self.retriever.index_documents(self.documents)
        with mock.patch.object(self.retriever, "_calculate_knn", return_value=[]):
            result = self.retriever.retrieve_relevant_documents(self.query, 2)
            self.assertIsNone(result)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_vectorized_ideal(self) -> None:
        """
        Ideal scenario for retrieve_vectorized file
        """
        query_vector = (
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
        )
        expected = (
            "Котёнок, которого мы нашли во дворе, очень забавный и пушистый. "
            "По утрам я играю с ним в догонялки перед работой."
        )
        self.retriever.index_documents(self.documents)
        actual = self.retriever.retrieve_vectorized(query_vector)
        self.assertEqual(actual, expected)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_vectorized_invalid_length(self):
        """
        Handling cases with difference in documents' and query's lengths
        """
        query_vector = (
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
            0.0,
        )
        self.retriever.index_documents(self.documents)
        actual = self.retriever.retrieve_vectorized(query_vector)
        self.assertIsNone(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_retrieve_vectorized_invalid_input(self) -> None:
        """
        Invalid input scenario for retrieve_vectorized method
        """
        bad_inputs = [[], {}, 0.7, 7, "", False]
        self.retriever.index_documents(self.documents)
        for bad_input in bad_inputs:
            actual = self.retriever.retrieve_vectorized(bad_input)
            self.assertIsNone(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_empty_knn_result_retrieve_vectorized(self):
        """
        Tests the case for retrieve_vectorized where _calculate_knn returns an empty list
        """
        query_vector = (
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
        )
        self.retriever.index_documents(self.documents)
        with mock.patch.object(self.retriever, "_calculate_knn", return_value=[]):
            result = self.retriever.retrieve_vectorized(query_vector)
            self.assertIsNone(result)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_dump_documents_ideal(self):
        """
        Ideal scenario for _dump_documents method
        """
        self.retriever.index_documents(self.documents)
        actual = self.retriever._dump_documents()
        self.assertEqual(self.dump, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_dump_documents_return_value(self):
        """
        Checks return type for _dump_documents method
        """
        self.retriever.index_documents(self.documents)
        actual = self.retriever._dump_documents()
        self.assertIsInstance(actual, dict)

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
        new_retriever = BasicSearchEngine(self._vectorizer, self._tokenizer)
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
                "engine": {"documents": []},
            }
            json.dump(state, f)

        loaded_retriever = BasicSearchEngine(self._vectorizer, self._tokenizer)
        self.assertFalse(loaded_retriever.load(str(self.test_path)))

        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {
                "engine": {"document_vectors": []},
            }
            json.dump(state, f)

        loaded_retriever = BasicSearchEngine(self._vectorizer, self._tokenizer)
        self.assertFalse(loaded_retriever.load(str(self.test_path)))

        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {}
            json.dump(state, f)

        loaded_retriever = BasicSearchEngine(self._vectorizer, self._tokenizer)
        self.assertFalse(loaded_retriever.load(str(self.test_path)))

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_return_value(self) -> None:
        """
        Checks return type for load method
        """
        self.retriever.index_documents(self.documents)
        actual = self.retriever.load(str(self.exp_path))
        self.assertIsInstance(actual, bool)
        self.assertTrue(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_documents_ideal(self) -> None:
        """
        Ideal scenario for loading file
        """
        self.retriever.index_documents(self.documents)
        new_retriever = BasicSearchEngine(self._vectorizer, self._tokenizer)
        new_retriever.load(str(self.exp_path))
        self.assertEqual(self.retriever._documents, new_retriever._documents)
        self.assertEqual(self.retriever._document_vectors, new_retriever._document_vectors)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_documents_missing_values(self) -> None:
        """
        Missing values scenario for loading documents
        """
        self.retriever.index_documents(self.documents)
        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {"documents": []}
            json.dump(state, f)

        loaded_retriever = BasicSearchEngine(self._vectorizer, self._tokenizer)
        self.assertFalse(loaded_retriever.load(str(self.test_path)))

        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {
                "document_vectors": [],
            }
            json.dump(state, f)

        loaded_retriever = BasicSearchEngine(self._vectorizer, self._tokenizer)
        self.assertFalse(loaded_retriever.load(str(self.test_path)))

        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {}
            json.dump(state, f)

        loaded_retriever = BasicSearchEngine(self._vectorizer, self._tokenizer)
        self.assertFalse(loaded_retriever.load(str(self.test_path)))

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_documents_return_value(self) -> None:
        """
        Checks return type for _load_documents method
        """
        self.retriever.index_documents(self.documents)
        actual = self.retriever.load(str(self.exp_path))
        self.assertIsInstance(actual, bool)
        self.assertTrue(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_documents_empty_load_vector(self) -> None:
        """
        Empty load_vector scenario for _load_documents
        """
        with mock.patch("lab_3_ann_retriever___vhkefbckemvkjo.main.load_vector", return_value=None):
            actual = self.retriever.load(str(self.exp_path))
        self.assertFalse(actual)
