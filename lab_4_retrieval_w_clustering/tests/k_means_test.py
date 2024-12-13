"""
Checks the fourth lab's KMeans class.
"""

# pylint: disable=protected-access, invalid-name
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
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
        self.mock_db.get_raw_documents.side_effect = lambda indices: [
            f"Document {i}" for i in indices
        ]
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
    def test_run_single_train_iteration_ideal(self):
        """
        Test run_single_train_iteration method.
        """
        cluster1 = MagicMock(spec=ClusterDTO)
        cluster1.get_centroid.return_value = (0.0, 0.0)
        cluster1.get_indices.return_value = [0]

        cluster2 = MagicMock(spec=ClusterDTO)
        cluster2.get_centroid.return_value = (1.0, 1.0)
        cluster2.get_indices.return_value = [1]

        setattr(self.kmeans, "_KMeans__clusters", [cluster1, cluster2])

        with patch("lab_4_retrieval_w_clustering.main.calculate_distance") as mock_distance:
            mock_distance.side_effect = lambda c, v: sum((ce - ve) ** 2 for ce, ve in zip(c, v))

            new_clusters = self.kmeans.run_single_train_iteration()

            self.assertEqual(len(new_clusters), 2)
            for cluster in new_clusters:
                self.assertIsInstance(cluster, ClusterDTO)
                cluster.set_new_centroid.assert_called_once()

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_run_single_train_iteration_distance_none(self):
        """
        Test run_single_train_iteration raises ValueError when calculate_distance returns None.
        """
        cluster = MagicMock(spec=ClusterDTO)
        cluster.get_centroid.return_value = (0.0, 0.0)
        setattr(self.kmeans, "_KMeans__clusters", [cluster])

        with patch("lab_4_retrieval_w_clustering.main.calculate_distance", return_value=None):
            with self.assertRaises(ValueError):
                self.kmeans.run_single_train_iteration()

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_run_single_train_iteration_updates_clusters(self):
        """
        Test run_single_train_iteration updates clusters correctly.
        """
        # Setup initial clusters
        cluster1 = MagicMock(spec=ClusterDTO)
        cluster1.get_centroid.return_value = (0.0, 0.0)
        cluster1.get_indices.return_value = []

        cluster2 = MagicMock(spec=ClusterDTO)
        cluster2.get_centroid.return_value = (1.0, 1.0)
        cluster2.get_indices.return_value = []

        setattr(self.kmeans, "_KMeans__clusters", [cluster1, cluster2])

        with patch("lab_4_retrieval_w_clustering.main.calculate_distance") as mock_distance:
            mock_distance.side_effect = lambda c, v: sum((ce - ve) ** 2 for ce, ve in zip(c, v))

            new_clusters = self.kmeans.run_single_train_iteration()

            self.assertEqual(len(new_clusters), 2)
            for cluster in new_clusters:
                self.assertIsInstance(cluster, ClusterDTO)
                cluster.set_new_centroid.assert_called_once()
                cluster.add_document_index.assert_called()

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
    def test_method_returns_none(self):
        """
        Test handling when a method returns None.
        """
        setattr(self.kmeans, "_KMeans__clusters", [ClusterDTO((0.0, 0.0))])

        with patch("lab_4_retrieval_w_clustering.main.calculate_distance", return_value=None):
            with self.assertRaises(ValueError):
                self.kmeans._is_convergence_reached([ClusterDTO((0.0, 0.0))])

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
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_empty_clusters(self):
        """
        Test scenario with no clusters.
        """
        setattr(self.kmeans, "_KMeans__clusters", [])

        with self.assertRaises(ValueError):
            self.kmeans._is_convergence_reached([])

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
        expected_vectors = [0.7071067811865476, 0.7071067811865476]
        expected_ids = [0, 2]
        for idx, pair in enumerate(result):
            self.assertEqual(expected_ids[idx], pair[1])
            np.testing.assert_almost_equal(pair[0], expected_vectors[idx])
        self.assertEqual(len(result), n_neighbours)
        self.assertIsInstance(result, list)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_infer_ideal_second_case(self):
        """
        Test inference method second scenario.
        """
        self.kmeans.train()
        query_vector = (0.5, 0.5)
        n_neighbours = 2

        with patch(
            "lab_4_retrieval_w_clustering.main.calculate_distance", side_effect=[0.3, 0.1, 0.9, 0.1]
        ):
            result = self.kmeans.infer(query_vector, n_neighbours)

        expected_vectors = [0.1, 0.9]
        expected_ids = [3, 1]
        for idx, pair in enumerate(result):
            self.assertEqual(expected_ids[idx], pair[1])
            np.testing.assert_almost_equal(pair[0], expected_vectors[idx])
        self.assertEqual(len(result), n_neighbours)
        self.assertIsInstance(result, list)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_infer_no_centroid(self):
        """
        Test inference method without centroid.
        """
        self.kmeans.train()
        query_vector = (0.5, 0.5)
        n_neighbours = 2
        for cluster in self.kmeans._KMeans__clusters[1:]:
            setattr(cluster, "_ClusterDTO__centroid", None)

        result = self.kmeans.infer(query_vector, n_neighbours)

        expected_vectors = [0.7071067811865476, 0.7071067811865476]
        expected_ids = [0, 2]
        for idx, pair in enumerate(result):
            self.assertEqual(expected_ids[idx], pair[1])
            np.testing.assert_almost_equal(pair[0], expected_vectors[idx])
        self.assertEqual(len(result), n_neighbours)
        self.assertIsInstance(result, list)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_infer_calculate_distance_return_none(self):
        """
        Test inference method.
        """
        self.kmeans.train()
        query_vector = (0.5, 0.5)
        n_neighbours = 2
        with patch("lab_4_retrieval_w_clustering.main.calculate_distance", return_value=None):
            with self.assertRaises(ValueError):
                self.kmeans.infer(query_vector, n_neighbours)

        with patch("lab_4_retrieval_w_clustering.main.calculate_distance", side_effect=[1.0, None]):
            with self.assertRaises(ValueError):
                self.kmeans.infer(query_vector, n_neighbours)

        with patch(
            "lab_4_retrieval_w_clustering.main.calculate_distance",
            side_effect=[1.0, 1.0, 1.0, None],
        ):
            with self.assertRaises(ValueError):
                self.kmeans.infer(query_vector, n_neighbours)

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
    def test_get_clusters_info_ideal(self):
        """
        Test get_clusters_info with valid inputs.
        """
        cluster1 = MagicMock(spec=ClusterDTO)
        cluster1.get_centroid.return_value = (0.0, 0.0)
        cluster1.get_indices.return_value = [0, 2]

        cluster2 = MagicMock(spec=ClusterDTO)
        cluster2.get_centroid.return_value = (1.0, 1.0)
        cluster2.get_indices.return_value = [1, 3]

        setattr(self.kmeans, "_KMeans__clusters", [cluster1, cluster2])

        with patch("lab_4_retrieval_w_clustering.main.calculate_distance") as mock_distance:
            mock_distance.side_effect = lambda c, v: sum((ce - ve) ** 2 for ce, ve in zip(c, v))

            num_examples = 2
            result = self.kmeans.get_clusters_info(num_examples)

            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["cluster_id"], 0)
            self.assertEqual(result[1]["cluster_id"], 1)
            self.assertListEqual(result[0]["documents"], ["Document 0", "Document 2"])
            self.assertListEqual(result[1]["documents"], ["Document 1", "Document 3"])

            num_examples = 1
            result = self.kmeans.get_clusters_info(num_examples)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["cluster_id"], 0)
            self.assertEqual(result[1]["cluster_id"], 1)
            self.assertListEqual(result[0]["documents"], ["Document 0"])
            self.assertListEqual(result[1]["documents"], ["Document 1"])

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark10
    def test_get_clusters_info_empty_clusters(self):
        """
        Test get_clusters_info with no clusters.
        """
        setattr(self.kmeans, "_KMeans__clusters", [])

        num_examples = 2
        result = self.kmeans.get_clusters_info(num_examples)

        # Assertions
        self.assertEqual(len(result), 0)
        setattr(self.kmeans, "_KMeans__clusters", [])
        with self.assertRaises(ValueError):
            self.kmeans._is_convergence_reached([])

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark10
    def test_get_clusters_info_distance_none(self):
        """
        Test get_clusters_info raises ValueError when calculate_distance returns None.
        """
        setattr(self.kmeans, "_KMeans__clusters", [ClusterDTO((0.0, 0.0))])

        self.kmeans._KMeans__clusters[0].get_indices = MagicMock(return_value=[0, 2])

        with patch("lab_4_retrieval_w_clustering.main.calculate_distance", return_value=None):
            with self.assertRaises(ValueError):
                self.kmeans.get_clusters_info(2)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark10
    def test_get_clusters_info_invalid_num_examples(self):
        """
        Test get_clusters_info with invalid num_examples.
        """
        with self.assertRaises(ValueError):
            self.kmeans.get_clusters_info(0)

        with self.assertRaises(ValueError):
            self.kmeans.get_clusters_info(-1)
