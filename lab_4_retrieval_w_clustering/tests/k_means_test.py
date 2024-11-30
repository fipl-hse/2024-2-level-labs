"""
Checks the fourth lab's KMeans class.
"""

# pylint: disable=protected-access, invalid-name
import unittest
from unittest.mock import MagicMock, patch

import pytest

from lab_4_retrieval_w_clustering.main import ClusterDTO, KMeans


class KMeansTest(unittest.TestCase):
    """
    Tests KMeans class functionality.
    """

    def setUp(self) -> None:
        self.mock_db = MagicMock()
        self.mock_db.get_vectors.side_effect = lambda indices=None: (
            [
                (0, (0.0, 0.0)),
                (1, (1.0, 1.0)),
                (2, (0.5, 0.5)),
                (3, (1.5, 1.5)),
            ]
            if indices is None
            else [(i, (i * 0.5, i * 0.5)) for i in indices]
        )

        self.kmeans = KMeans(self.mock_db, n_clusters=2)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_initialization_ideal(self):
        """
        Test initialization of KMeans instance.
        """
        self.assertEqual(self.kmeans._n_clusters, 2)
        self.assertEqual(self.kmeans._db, self.mock_db)
        self.assertEqual(len(self.kmeans._KMeans__clusters), 0)

    @patch("lab_4_retrieval_w_clustering.main.ClusterDTO")
    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_train_ideal(self, MockClusterDTO):
        """
        Test the train method for convergence.
        """
        mock_cluster = MagicMock()
        mock_cluster.get_centroid.return_value = (0.25, 0.25)
        MockClusterDTO.return_value = mock_cluster

        self.kmeans.train()
        self.assertEqual(len(self.kmeans._KMeans__clusters), 2)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_is_convergence_reached_ideal(self):
        """
        Test convergence checking.
        """
        setattr(self.kmeans, "_KMeans__clusters", [ClusterDTO((0.0, 0.0)), ClusterDTO((1.0, 1.0))])
        new_clusters = [ClusterDTO((0.01, 0.01)), ClusterDTO((1.01, 1.01))]

        self.assertTrue(self.kmeans._is_convergence_reached(self.kmeans._KMeans__clusters))
        self.assertFalse(self.kmeans._is_convergence_reached(new_clusters))

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_infer_ideal(self):
        """
        Test inference method.
        """
        self.kmeans.train()
        query_vector = (0.5, 0.5)
        n_neighbours = 2
        result = self.kmeans.infer(query_vector, n_neighbours)

        self.assertEqual(len(result), n_neighbours)
        self.assertIsInstance(result, list)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_infer_invalid_input(self):
        """
        Test inference with invalid inputs.
        """
        with self.assertRaises(ValueError):
            self.kmeans.infer((), 1)

        with self.assertRaises(ValueError):
            self.kmeans.infer((0.5, 0.5), 0)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark10
    def test_calculate_square_sum_ideal(self):
        """
        Test calculation of the square sum of distances.
        """
        self.kmeans.train()
        square_sum = self.kmeans.calculate_square_sum()

        self.assertIsInstance(square_sum, float)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark10
    def test_empty_clusters(self):
        """
        Test scenario with no clusters.
        """
        setattr(self.kmeans, "_KMeans__clusters", [])
        with self.assertRaises(ValueError):
            self.kmeans._is_convergence_reached([])

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark10
    def test_method_returns_none(self):
        """
        Test handling when a method returns None.
        """
        setattr(self.kmeans, "_KMeans__clusters", [ClusterDTO((0.0, 0.0))])
        with patch("lab_4_retrieval_w_clustering.main.calculate_distance", return_value=None):
            with self.assertRaises(ValueError):
                self.kmeans._is_convergence_reached([ClusterDTO((0.0, 0.0))])