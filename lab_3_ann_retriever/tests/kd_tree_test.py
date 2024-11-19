# pylint: disable=duplicate-code, protected-access
"""
Checks the third lab's КD Tree class.
"""

import unittest
from unittest import mock

import numpy as np
import pytest

from lab_3_ann_retriever.main import KDTree


class KDTreeTest(unittest.TestCase):
    """
    Tests KDTree class functionality.
    """

    def setUp(self) -> None:
        self.kdtree = KDTree()
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
    @pytest.mark.mark10
    def test_find_closest_ideal(self):
        """
        Ideal scenario for _find_closest method
        """
        expected = [(161.88301532908295, 7)]
        self.kdtree.build(self.points)
        actual = self.kdtree._find_closest(self.point)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_find_closest_invalid_input(self):
        """
        Invalid input scenario for _find_closest method
        """
        expected = None
        self.kdtree.build(self.points)
        bad_inputs = [(), None, {}, [], ""]
        for bad_input in bad_inputs:
            actual = self.kdtree._find_closest(self.point, bad_input)
            self.assertEqual(expected, actual)

            actual = self.kdtree._find_closest(bad_input, bad_input)
            self.assertEqual(expected, actual)

            actual = self.kdtree._find_closest(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_ann_retriever
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
    @pytest.mark.mark10
    def test_find_closest_empty_distances(self):
        """
        Empty distance scenario
        """
        self.kdtree.build(self.points)
        with mock.patch("lab_3_ann_retriever.main.calculate_distance", return_value=None):
            result = self.kdtree._find_closest(self.point)
        self.assertIsNone(result)

    @pytest.mark.lab_3_ann_retriever
    @pytest.mark.mark10
    def test_find_closest_away_node_is_worthy(self):
        """
        Away node is worth considering as the closest neighbour scenario
        """
        query = (-0.1, -0.2)
        vectors = [(-0.26, -0.26), (0.0, -0.51), (-0.51, 0.0), (0.0, -0.26), (0.0, 0.0)]

        self.kdtree.build(vectors)
        actual = self.kdtree._find_closest(query)

        np.testing.assert_almost_equal(actual, [(0.11661903789690602, 3)])
