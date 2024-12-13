# pylint: disable=protected-access
"""
Checks the third lab's Tokenizer class
"""
import json
import unittest
from pathlib import Path
from unittest import mock

import pytest

from config.constants import PROJECT_ROOT
from lab_3_ann_retriever___vhkefbckemvkjo.main import Tokenizer


def get_tokens_assets() -> tuple[list[str], list[str]]:
    """
    Get tokens assets.

    Returns:
         tuple[list[str], list[str]]: Assets for tests.
    """
    path = PROJECT_ROOT / "lab_3_ann_retriever___vhkefbckemvkjo" / "tests" / "assets" / "tokens.json"
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["tokens"], data["sorted_tokens"]


def get_documents_assets() -> tuple[list[str], list[str]]:
    """
    Get documents assets.

    Returns:
         tuple[list[str], list[str]]: Assets for tests.
    """
    path = PROJECT_ROOT / "lab_3_ann_retriever___vhkefbckemvkjo" / "tests" / "assets" / "documents.json"
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["documents"], data["tokenized_documents"]


class TokenizerTest(unittest.TestCase):
    """
    Tests Tokenizer class functionality
    """

    def setUp(self) -> None:
        path_to_test_directory = Path(__file__).parent.parent
        with open(
            path_to_test_directory / "assets" / "stopwords.txt", "r", encoding="utf-8"
        ) as stop_words:
            self._stop_words = [line.strip() for line in stop_words]
        self.tokenizer = Tokenizer(self._stop_words)
        self.tokens, self.expected_tokens = get_tokens_assets()
        self.documents, self.expected_documents = get_documents_assets()

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_remove_stop_words_ideal(self):
        """
        Ideal stopwords removing scenario
        """
        actual = self.tokenizer._remove_stop_words(self.tokens)
        self.assertEqual(self.expected_tokens, actual, "Stopwords were not removed!")

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_remove_stop_words_invalid_input(self):
        """
        Invalid input scenario in removing stopwords
        """
        expected = None
        bad_inputs = [None, True, 17, 3.14, (), {}, []]

        for bad_input in bad_inputs:
            self.assertEqual(expected, self.tokenizer._remove_stop_words(bad_input))

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_remove_stop_words_return_type(self):
        """
        Checks return type for removing stopwords
        """
        actual = self.tokenizer._remove_stop_words(
            ["мой", "кот", "вектор", "по", "утрам", "приносит", "мне", "тапочки"]
        )
        self.assertIsInstance(actual, list)
        for word in actual:
            self.assertIsInstance(word, str)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_ideal(self):
        """
        Ideal tokenize scenario
        """
        text = (
            "Мой кот Вектор по утрам приносит мне тапочки, "
            "а по вечерам мы гуляем с ним на шлейке во дворе. "
            "Вектор забавный и храбрый. Он не боится собак!"
        )
        actual = self.tokenizer.tokenize(text)
        self.assertEqual(self.expected_tokens, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_invalid_input(self):
        """
        Invalid input scenario in tokenization
        """
        bad_inputs = [[], {}, (), None, 7, 0.7, True]
        expected = None
        for bad_input in bad_inputs:
            actual = self.tokenizer.tokenize(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_remove_stopwords_none(self):
        """
        Checks Tokenizer tokenize method with None as _remove_stop_words return value
        """
        expected = None
        with mock.patch.object(self.tokenizer, "_remove_stop_words", return_value=None):
            actual = self.tokenizer.tokenize("Мой кот Вектор по утрам приносит мне тапочки.")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_return_value(self):
        """
        Checks return type for tokenization
        """
        actual = self.tokenizer.tokenize("Мой кот Вектор по утрам приносит мне тапочки.")
        self.assertIsInstance(actual, list)
        for word in actual:
            self.assertIsInstance(word, str)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_documents_ideal(self):
        """
        Ideal documents' tokenization scenario
        """
        actual = self.tokenizer.tokenize_documents(self.documents)
        self.assertEqual(self.expected_documents, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_documents_invalid_input(self):
        """
        Invalid input scenario for documents' tokenization
        """
        bad_inputs = [{}, (), None, 7, 0.7, True, ""]
        expected = None
        for bad_input in bad_inputs:
            actual = self.tokenizer.tokenize_documents(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_documents_none_tokenize(self):
        """
        Checks Tokenizer tokenize_documents method with None as tokenize return value
        """
        expected = None
        with mock.patch.object(self.tokenizer, "tokenize", return_value=None):
            actual = self.tokenizer.tokenize_documents(
                [
                    "Мой кот Вектор по утрам ",
                    "приносит мне тапочки.",
                    "Котёнок, которого мы нашли во дворе, ",
                    "очень забавный и ",
                    "пушистый.",
                    "Вектора используются для поиска ",
                    "релевантного документа.",
                ]
            )
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_documents_return_value(self):
        """
        Checks return value of documents' tokenization
        """
        actual = self.tokenizer.tokenize_documents(
            [
                "Мой кот Вектор по утрам приносит мне тапочки.",
                "Котёнок, которого мы нашли во дворе,",
                "очень забавный и пушистый.",
                "Вектора используются для поиска",
                " релевантного документа.",
            ]
        )
        self.assertIsInstance(actual, list)
        for new_list in actual:
            self.assertIsInstance(new_list, list)
            for word in new_list:
                self.assertIsInstance(word, str)
