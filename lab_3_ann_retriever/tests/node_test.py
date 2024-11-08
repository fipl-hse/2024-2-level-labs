# pylint: disable=duplicate-code
"""
Checks the third lab's Node class.
"""

import unittest
from unittest import mock

import pytest

from lab_3_ann_retriever.main import Node


class NodeTest(unittest.TestCase):
    """
    Tests Node class functionality
    """

    def setUp(self) -> None:
        self.node = Node((0.0, 0.0), -1, None, Node((0.1, 0.1), 0))
        self.state = {
            "vector": {"elements": {}, "len": 2},
            "left_node": None,
            "payload": -1,
            "right_node": {
                "left_node": None,
                "payload": 0,
                "right_node": None,
                "vector": {"elements": {0: 0.1, 1: 0.1}, "len": 2},
            },
        }
        self.right_node = Node()
        self.left_node = Node()

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_ideal(self):
        """
        Ideal scenario for save method
        """
        actual = self.node.save()
        self.assertEqual(self.state, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_ideal(self):
        """
        Ideal scenario for load method
        """
        actual = self.node.load(self.state)
        self.assertTrue(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_missing_vector(self):
        """
        Handling case where vector is missing
        """
        state = {
            "left_node": None,
            "payload": -1,
            "right_node": {
                "left_node": None,
                "payload": 0,
                "right_node": None,
                "vector": {"elements": {0: 0.1, 1: 0.1}, "len": 2},
            },
        }
        actual = self.node.load(state)
        self.assertFalse(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_missing_payload(self):
        """
        Handling case where payload is missing
        """
        state = {
            "left_node": None,
            "right_node": {
                "left_node": None,
                "payload": 0,
                "right_node": None,
                "vector": {"elements": {0: 0.1, 1: 0.1}, "len": 2},
            },
        }
        actual = self.node.load(state)
        self.assertFalse(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_invalid_vector_type(self):
        """
        Handling cases with wrong vector's type
        """
        invalid_vectors = [[], (), "who is vector?", 7, 0.7, False]
        for invalid_vector in invalid_vectors:

            state = {
                "left_node": None,
                "payload": -1,
                "right_node": {
                    "left_node": None,
                    "payload": 0,
                    "right_node": None,
                    "vector": {"elements": {0: 0.1, 1: 0.1}, "len": 2},
                },
                "vector": invalid_vector,
            }
            actual = self.node.load(state)
            self.assertFalse(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_invalid_payload(self):
        """
        Handling cases where payload is wrong
        """
        invalid_payloads = [[], 0.7, {}, (), False, ""]
        for invalid_payload in invalid_payloads:
            state = {
                "left_node": None,
                "payload": invalid_payload,
                "right_node": {
                    "left_node": None,
                    "payload": 0,
                    "right_node": None,
                    "vector": {"elements": {0: 0.1, 1: 0.1}, "len": 2},
                },
            }
            actual = self.node.load(state)
            self.assertFalse(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_return_value(self):
        """
        Checks return type for load method
        """
        actual = self.node.load(self.state)
        self.assertIsInstance(actual, bool)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_none_load_vector(self):
        """
        Handling the case where load_vector returns None
        """
        with mock.patch("lab_3_ann_retriever.main.load_vector", return_value=None):
            result = self.node.load(self.state)
        self.assertFalse(result)
