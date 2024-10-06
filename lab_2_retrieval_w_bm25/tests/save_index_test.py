""""
Checks the second lab's index saving function
"""
import json
import shutil
import unittest
from pathlib import Path

import pytest

from lab_2_retrieval_w_bm25.main import save_index


class SaveIndexTest(unittest.TestCase):
    """
    Tests for index saving functions
    """
    def setUp(self) -> None:
        self.test_path = Path(__file__).parent.parent / "test_tmp" / "saved_indexes.txt"
        self.test_path.parent.mkdir(parents=True, exist_ok=True)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_save_index_ideal(self):
        """
        Ideal scenario
        """
        bm25_with_cutoff_new_document_score = {
            'string': 0.0, 'ghost': 0.0, 'named': 0.0, 'saw': 0.5526,
            'beautiful': 0.3664, 'particular': 0.0, 'reading': 0.5526,
            'library': 0.5522, 'kind': 0.5526, 'opened': 0.0, 'boy': 0.5526,
            'street': 0.3664, 'must': 0.0, 'need': 0.0, 'lived': 0.5526,
            'smart': 0.5526, 'began': 0.0, 'friend': 0.9871, 'coffee': 0.9871,
            'upon': 0.5526, 'time': 0.5526, 'love': 0.0, 'writing': 0.0,
            'format': 0.0, 'story': 0.0, 'cat': 0.4075, 'function': 0.0,
            'loved': 0.9871, 'tale': 0.0, 'told': 0.0, 'thought': 0.0,
            'shop': 0.9871, 'ugly': 0.0, 'output': 0.0, 'across': 0.3664
        }

        save_index([bm25_with_cutoff_new_document_score], str(self.test_path))
        with open(str(self.test_path), 'r', encoding='utf-8') as f:
            saved_indexes = json.load(f)

        with open(Path(__file__).parent / "assets" / "saved_metrics_example.json",
                  'r', encoding='utf-8') as f:
            example_metrics = json.load(f)

        for (proper_line, student_line) in zip(example_metrics[0], saved_indexes[0]):
            self.assertEqual(proper_line, student_line)

    @pytest.mark.lab_2_retrieval_w_bm25
    @pytest.mark.mark10
    def test_save_index_bad_input(self):
        """
        Bad input scenario
        """
        shutil.rmtree(Path(__file__).parent.parent / "test_tmp")
        bad_path_inputs = [None, True, 42, 3.14, (), {}, [], '']
        bad_index_inputs = [None, True, 42, 3.14, 'string', (), {}, [], {1: 3, 5: 'seven'},
                            {'token': {'not': 'frequency'}}, {'bad frequency type': 10}]

        for bad_path in bad_path_inputs:
            save_index([{'token': 1.0}], bad_path)
        for bad_index in bad_index_inputs:
            save_index(bad_index, 'something/resembling/a/path')

        self.assertFalse(self.test_path.is_file())

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Define final instructions for ArticleInstanceCreationBasicTest class.
        """
        test_dir = Path(__file__).parent.parent / 'test_tmp'
        if test_dir.is_dir():
            shutil.rmtree(test_dir)
