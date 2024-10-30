# pylint: disable=protected-access
"""
Checks the third lab's Vectorizer class.
"""

import unittest

import numpy as np
import pytest

from lab_3_ann_retriever.main import Vectorizer


class VectorizerTest(unittest.TestCase):
    """
    Tests Vectorizer class functionality.
    """

    def setUp(self) -> None:

        self._vocabulary = [
            "боится",
            "вектор",
            "вектора",
            "вечерам",
            "вместе",
            "гуляем",
            "давайте",
            "дворе",
            "догонялки",
            "документа",
            "думает",
            "забавно",
            "забавный",
            "играю",
            "использовать",
            "используются",
            "кот",
            "которого",
            "котёнок",
            "кошка",
            "любимый",
            "мягкий",
            "наблюдать",
            "научимся",
            "нашли",
            "очень",
            "плед",
            "поиска",
            "приносит",
            "просто",
            "пушистый",
            "работой",
            "релевантного",
            "собак",
            "собака",
            "создавать",
            "спят",
            "тапочки",
            "утрам",
            "храбрый",
            "шлейке",
            "это",
        ]
        corpus = [
            [
                "вектора",
                "используются",
                "поиска",
                "релевантного",
                "документа",
                "давайте",
                "научимся",
                "создавать",
                "использовать",
            ],
            [
                "кот",
                "вектор",
                "утрам",
                "приносит",
                "тапочки",
                "вечерам",
                "гуляем",
                "шлейке",
                "дворе",
                "вектор",
                "забавный",
                "храбрый",
                "боится",
                "собак",
            ],
            [
                "котёнок",
                "которого",
                "нашли",
                "дворе",
                "очень",
                "забавный",
                "пушистый",
                "утрам",
                "играю",
                "догонялки",
                "работой",
            ],
            [
                "собака",
                "думает",
                "любимый",
                "плед",
                "это",
                "кошка",
                "просто",
                "очень",
                "пушистый",
                "мягкий",
                "забавно",
                "наблюдать",
                "спят",
                "вместе",
            ],
        ]
        self._idf_values = {
            "боится": 0.8472978603872037,
            "вектор": 0.8472978603872037,
            "вектора": 0.8472978603872037,
            "вечерам": 0.8472978603872037,
            "вместе": 0.8472978603872037,
            "гуляем": 0.8472978603872037,
            "давайте": 0.8472978603872037,
            "дворе": 0.0,
            "догонялки": 0.8472978603872037,
            "документа": 0.8472978603872037,
            "думает": 0.8472978603872037,
            "забавно": 0.8472978603872037,
            "забавный": 0.0,
            "играю": 0.8472978603872037,
            "использовать": 0.8472978603872037,
            "используются": 0.8472978603872037,
            "кот": 0.8472978603872037,
            "которого": 0.8472978603872037,
            "котёнок": 0.8472978603872037,
            "кошка": 0.8472978603872037,
            "любимый": 0.8472978603872037,
            "мягкий": 0.8472978603872037,
            "наблюдать": 0.8472978603872037,
            "научимся": 0.8472978603872037,
            "нашли": 0.8472978603872037,
            "очень": 0.0,
            "плед": 0.8472978603872037,
            "поиска": 0.8472978603872037,
            "приносит": 0.8472978603872037,
            "просто": 0.8472978603872037,
            "пушистый": 0.0,
            "работой": 0.8472978603872037,
            "релевантного": 0.8472978603872037,
            "собак": 0.8472978603872037,
            "собака": 0.8472978603872037,
            "создавать": 0.8472978603872037,
            "спят": 0.8472978603872037,
            "тапочки": 0.8472978603872037,
            "утрам": 0.0,
            "храбрый": 0.8472978603872037,
            "шлейке": 0.8472978603872037,
            "это": 0.8472978603872037,
        }

        self._token_to_ids = {
            "боится": 0,
            "вектор": 1,
            "вектора": 2,
            "вечерам": 3,
            "вместе": 4,
            "гуляем": 5,
            "давайте": 6,
            "дворе": 7,
            "догонялки": 8,
            "документа": 9,
            "думает": 10,
            "забавно": 11,
            "забавный": 12,
            "играю": 13,
            "использовать": 14,
            "используются": 15,
            "кот": 16,
            "которого": 17,
            "котёнок": 18,
            "кошка": 19,
            "любимый": 20,
            "мягкий": 21,
            "наблюдать": 22,
            "научимся": 23,
            "нашли": 24,
            "очень": 25,
            "плед": 26,
            "поиска": 27,
            "приносит": 28,
            "просто": 29,
            "пушистый": 30,
            "работой": 31,
            "релевантного": 32,
            "собак": 33,
            "собака": 34,
            "создавать": 35,
            "спят": 36,
            "тапочки": 37,
            "утрам": 38,
            "храбрый": 39,
            "шлейке": 40,
            "это": 41,
        }
        self.vectorizer = Vectorizer(corpus)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_vocabulary_ideal(self):
        """
        Ideal build_vocabulary scenario
        """
        self.vectorizer.build()

        self.assertEqual(self._vocabulary, self.vectorizer._vocabulary)
        self.assertEqual(self._idf_values, self.vectorizer._idf_values)
        self.assertEqual(self._token_to_ids, self.vectorizer._token2ind)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_vocabulary_invalid_input(self):
        """
        Invalid input build_vocabulary scenario
        """
        bad_inputs = [None, {}, []]
        for bad_input in bad_inputs:
            vectorizer = Vectorizer(bad_input)
            self.assertRaises(ValueError, vectorizer.build)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_idf_ideal(self):
        """
        Ideal _calculate_tf_idf scenario
        """
        self.vectorizer.build()
        document = ["кот", "вектор", "кот"]
        expected = [
            0.0,
            0.2824326201290679,
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
            0.0,
            0.0,
            0.0,
            0.5648652402581358,
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
            0.0,
            0.0,
            0.0,
        ]
        actual = self.vectorizer._calculate_tf_idf(document)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_idf_invalid_input(self):
        """
        Invalid input _calculate_tf_idf scenario
        """
        expected = None
        bad_inputs = [None, True, 42, 3.14, (), "частотность", [], {}]
        for bad_input in bad_inputs:
            actual = self.vectorizer._calculate_tf_idf(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_idf_return_value(self):
        """
        Checks return type for _calculate_tf_idf
        """
        self.vectorizer.build()
        document = ["кот", "вектор", "кот"]
        actual = self.vectorizer._calculate_tf_idf(document)
        self.assertIsInstance(actual, tuple)
        for freq in actual:
            self.assertIsInstance(freq, float)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_vectorizer_ideal(self):
        """
        Ideal vectorization scenario
        """
        tokenized_document = [
            "кот",
            "вектор",
            "утрам",
            "приносит",
            "тапочки",
            "вечерам",
            "гуляем",
            "шлейке",
            "дворе",
            "вектор",
            "забавный",
            "храбрый",
            "боится",
            "собак",
        ]
        expected = [
            0.06052127574194312,
            0.12104255148388623,
            0.0,
            0.06052127574194312,
            0.0,
            0.06052127574194312,
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
            0.06052127574194312,
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
            0.06052127574194312,
            0.0,
            0.0,
            0.0,
            0.0,
            0.06052127574194312,
            0.0,
            0.0,
            0.0,
            0.06052127574194312,
            0.0,
            0.06052127574194312,
            0.06052127574194312,
            0.0,
        ]
        self.vectorizer.build()
        actual = self.vectorizer.vectorize(tokenized_document)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_vectorizer_invalid_input(self):
        """
        Invalid input vectorization scenario
        """
        expected = None
        bad_inputs = [None, True, 42, 3.14, (), "", [], {}]
        for bad_input in bad_inputs:
            actual = self.vectorizer.vectorize(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_vectorizer_return_value(self):
        """
        Checks return type for vectorization
        """
        tokenized_document = [
            "кот",
            "вектор",
            "утрам",
            "приносит",
            "тапочки",
            "вечерам",
            "гуляем",
            "шлейке",
            "дворе",
            "вектор",
            "забавный",
            "храбрый",
            "боится",
            "собак",
        ]
        actual = self.vectorizer.vectorize(tokenized_document)
        self.assertIsInstance(actual, tuple)
        for freq in actual:
            self.assertIsInstance(freq, float)
