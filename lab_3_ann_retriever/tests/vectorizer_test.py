# pylint: disable=protected-access
"""
Checks the third lab's Vectorizer class.
"""
import json
import unittest
from pathlib import Path

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
        self.corpus = [
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
        self.vectorizer = Vectorizer(self.corpus)
        self.exp_path = Path(__file__).parent.parent / "tests" / "assets" / "vectorizer_data.json"
        self.test_path = Path(__file__).parent.parent / "test_tmp" / "vectorizer_state.json"
        self.test_path.parent.mkdir(parents=True, exist_ok=True)

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
        bad_inputs = [None, [], {}]
        for bad_input in bad_inputs:
            vectorizer = Vectorizer(bad_input)
            self.assertFalse(vectorizer.build())

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

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_vector2tokens_ideal(self):
        """
        Ideal scenario for vector2tokens method
        """
        expected = [
            "вместе",
            "думает",
            "забавно",
            "кошка",
            "любимый",
            "мягкий",
            "наблюдать",
            "плед",
            "просто",
            "собака",
            "спят",
            "это",
        ]
        vector = (
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
        )
        self.vectorizer.build()
        actual = self.vectorizer.vector2tokens(vector)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_vector2tokens_wrong_length(self):
        """
        Wrong length scenario for vector2tokens method
        """
        vector = (
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
        )
        self.vectorizer.build()
        actual = self.vectorizer.vector2tokens(vector)
        self.assertIsNone(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_vector2tokens_return_value(self):
        """
        Checks return type for vector2tokens method
        """
        vector = (
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
        )
        self.vectorizer.build()
        actual = self.vectorizer.vector2tokens(vector)
        self.assertIsInstance(actual, list)
        for word in actual:
            self.assertIsInstance(word, str)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_ideal(self):
        """
        Ideal scenario for save method
        """
        self.vectorizer.build()
        self.vectorizer.save(str(self.test_path))

        with open(str(self.test_path), "r", encoding="utf-8") as f:
            actual = json.load(f)

        with open(str(self.exp_path), "r", encoding="utf-8") as f:
            expected = json.load(f)
        self.assertEqual(actual, expected)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_return_value(self):
        """
        Checks return type for save method
        """
        self.vectorizer.build()
        self.assertIsInstance(self.vectorizer.save(str(self.test_path)), bool)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_ideal(self) -> None:
        """
        Ideal scenario for loading file
        """
        self.vectorizer.build()
        new_vectorizer = Vectorizer(self.corpus)
        new_vectorizer.load(str(self.exp_path))
        self.assertEqual(self.vectorizer._idf_values, new_vectorizer._idf_values)
        self.assertEqual(self.vectorizer._vocabulary, new_vectorizer._vocabulary)
        self.assertEqual(self.vectorizer._token2ind, new_vectorizer._token2ind)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_missing_values(self) -> None:
        """
        Missing values scenario for loading file
        """
        self.vectorizer.build()
        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {
                "vocabulary": self.vectorizer._vocabulary,
                "token2ind": self.vectorizer._token2ind,
            }
            json.dump(state, f)

        loaded_vectorizer = Vectorizer(self.corpus)
        self.assertFalse(loaded_vectorizer.load(str(self.test_path)))

        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {
                "vocabulary": self.vectorizer._vocabulary,
                "idf_values": self.vectorizer._idf_values,
            }
            json.dump(state, f)

        loaded_vectorizer = Vectorizer(self.corpus)
        self.assertFalse(loaded_vectorizer.load(str(self.test_path)))

        with open(str(self.test_path), "w", encoding="utf-8") as f:
            state = {
                "idf_values": self.vectorizer._idf_values,
                "token2ind": self.vectorizer._token2ind,
            }
            json.dump(state, f)

        loaded_vectorizer = Vectorizer(self.corpus)
        self.assertFalse(loaded_vectorizer.load(str(self.test_path)))

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_return_value(self) -> None:
        """
        Checks return type for load method
        """
        self.vectorizer.build()
        actual = self.vectorizer.load(str(self.exp_path))
        self.assertIsInstance(actual, bool)
        self.assertTrue(actual)