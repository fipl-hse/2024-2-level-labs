# pylint: disable=protected-access
"""
Checks the third lab's Naive KDTree class.
"""

import unittest
from unittest import mock


import pytest

import lab_3_ann_retriever.main
from lab_3_ann_retriever.main import NaiveKDTree, Node


class NaiveKDTreeTest(unittest.TestCase):
    """
    Tests NaiveKDTree class functionality.
    """

    def setUp(self) -> None:
        self.kdtree = NaiveKDTree()
        self.points = [
            (63.63961, 144.23502),
            (222.23357, 385.66147),
            (545.48236, 146.25533),
            (645.48749, 502.83917),
            (1043.4875, 168.47868),
            (1494.0156, 590.72247),
            (1096.0155, 661.43311),
            (803.07129, 323.03201),
            (358.60416, 654.36206),
            (510.12704, 321.01172),
            (852.56873, 677.59558),
        ]
        self.point = (953.58398, 382.63101)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_fields(self):
        """
        Check fields of class are created correctly
        """
        self.assertIsNone(NaiveKDTree()._root)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_ideal(self):
        """
        Ideal scenario for build method
        """
        expected = True
        actual = self.kdtree.build(self.points)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_invalid_input(self):
        """
        Invalid input scenario for build method
        """
        expected = False
        bad_inputs = [[], {}, (), "", None]
        for bad_input in bad_inputs:
            actual = self.kdtree.build(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_return_value(self):
        """
        Checks return type for build method
        """
        actual = self.kdtree.build(self.points)
        self.assertIsInstance(actual, bool)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_query_ideal(self):
        """
        Ideal scenario for query method
        """
        self.kdtree.build(self.points)
        actual = self.kdtree._find_closest(self.point, 1)

        expected_neighbor = [(161.88301532908295, 7)]
        self.assertEqual(expected_neighbor, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_query_invalid_input(self):
        """
        Invalid input scenario for query method
        """
        expected = None
        bad_inputs = [[], {}, 0.7, (), "", None]
        for bad_input in bad_inputs:
            actual = self.kdtree.query(bad_input)
            self.assertEqual(expected, actual)

            actual = self.kdtree.query(bad_input, bad_input)
            self.assertEqual(expected, actual)

            actual = self.kdtree.query(self.point, bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_query_return_value(self):
        """
        Checks return type for query method
        """
        self.kdtree.build(self.points)
        actual = self.kdtree._find_closest(self.point, 1)
        self.assertIsInstance(actual, list)
        for ind in actual:
            self.assertIsInstance(ind, tuple)
            self.assertIsInstance(ind[0], float)
            self.assertIsInstance(ind[1], int)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_closest_ideal(self):
        """
        Ideal scenario for _find_closest method
        """
        expected = [(161.88301532908295, 7)]
        self.kdtree.build(self.points)
        actual = self.kdtree._find_closest(self.point)
        np.testing.assert_almost_equal(actual[0][0], expected[0][0])
        self.assertEqual(actual[0][1], expected[0][1])

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_closest_invalid_input(self):
        """
        Invalid input scenario for _find_closest method
        """
        expected = None
        self.kdtree.build(self.points)
        bad_inputs = [(), None, {}, 0.7, [], ""]
        for bad_input in bad_inputs:
            actual = self.kdtree._find_closest(self.point, bad_input)
            self.assertEqual(expected, actual)

            actual = self.kdtree._find_closest(bad_input, bad_input)
            self.assertEqual(expected, actual)

            actual = self.kdtree._find_closest(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_closest_return_value(self):
        """
        Checks return type for _find_closest method
        """
        self.kdtree.build(self.points)
        actual = self.kdtree._find_closest(self.point)
        self.assertIsInstance(actual, list)
        for neighbour in actual:
            self.assertIsInstance(neighbour, tuple)
            self.assertIsInstance(neighbour[0], float)
            self.assertIsInstance(neighbour[1], int)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_query_none_knn(self):
        """
        Scenario of _find_closest returning None
        """
        with mock.patch.object(self.kdtree, "_find_closest", return_value=None):
            actual = self.kdtree.query(self.point)
        self.assertIsNone(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_closest_none_distance(self):
        """
        Checks return of _find_closest method with None calculate_distance
        """
        self.kdtree.build(self.points)
        with mock.patch.object(lab_3_ann_retriever.main, "calculate_distance", return_value=None):
            actual = self.kdtree._find_closest(self.point)
        self.assertIsNone(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_ideal(self):
        """
        Ideal scenario for save method
        """
        expected = {
            "root": {
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
        }
        node = Node((0.0, 0.0), -1, None, Node((0.1, 0.1), 0))
        self.kdtree._root = node
        actual = self.kdtree.save()
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_missing_root(self):
        """
        Handles missing root case
        """
        self.kdtree._root = None
        self.assertIsNone(self.kdtree.save())

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_save_return_value(self):
        """
        Checks return type for save_method
        """
        node = Node((0.0, 0.0), -1, None, Node((0.1, 0.1), 0))
        self.kdtree._root = node
        actual = self.kdtree.save()
        self.assertIsInstance(actual, dict)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_ideal(self):
        """
        Ideal scenario for load method
        """
        state = {
            "root": {
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
        }
        node = Node((0.0, 0.0), -1, None, Node((0.1, 0.1), 0))
        self.kdtree._root = node
        actual = self.kdtree.load(state)
        self.assertTrue(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_missing_root(self):
        """
        Handling case with missing root
        """
        state = {
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

        node = Node((0.0, 0.0), -1, None, Node((0.1, 0.1), 0))
        self.kdtree._root = node
        actual = self.kdtree.load(state)
        self.assertFalse(actual)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_load_return_value(self):
        """
        Checks return type for load method
        """
        state = {
            "root": {
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
        }
        node = Node((0.0, 0.0), -1, None, Node((0.1, 0.1), 0))
        self.kdtree._root = node
        actual = self.kdtree.load(state)
        self.assertIsInstance(actual, bool)
