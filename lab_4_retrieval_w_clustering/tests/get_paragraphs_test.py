"""
Checks the fourth lab's get_paragraphs function.
"""

import unittest

import pytest

from lab_4_retrieval_w_clustering.main import get_paragraphs


class GetParagraphsTest(unittest.TestCase):
    """
    Tests GetParagraphsTest class functionality.
    """

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_paragraphs_ideal(self):
        """
        Ideal get_paragraphs scenario
        """
        text = "Привет!\nЭто тестовый запрос в функцию!\nСпасибо, что прочитал это!\nВсех благ!"
        actual = get_paragraphs(text)
        expected = [
            "Привет!",
            "Это тестовый запрос в функцию!",
            "Спасибо, что прочитал это!",
            "Всех благ!",
        ]
        self.assertEqual(expected, actual)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_paragraphs_fails(self):
        """
        Scenario when get_paragraphs raises ValueError because of empty input
        """
        self.assertRaises(ValueError, get_paragraphs, "")
