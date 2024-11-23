# pylint: disable=protected-access
"""
Checks the fourth lab's ClusterDTO class.
"""
import unittest

import pytest

from lab_4_retrieval_w_clustering.main import ClusterDTO


class ClusterDTOTest(unittest.TestCase):
    """
    Tests ClusterDTO class functionality.
    """

    def setUp(self) -> None:
        self.centroid = (0.5, 0.5, 0.5)
        self.cluster_dto = ClusterDTO(self.centroid)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_centroid_initial(self):
        """
        Test initial centroid assignment.
        """
        self.assertEqual(self.cluster_dto.get_centroid(), self.centroid)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_set_new_centroid_ideal(self):
        """
        Test setting a new centroid with valid input.
        """
        new_centroid = (1.0, 1.0, 1.0)
        self.cluster_dto.set_new_centroid(new_centroid)
        self.assertEqual(self.cluster_dto.get_centroid(), new_centroid)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_set_new_centroid_empty_input(self):
        """
        Test setting a new centroid with empty input.
        """
        with self.assertRaises(ValueError):
            self.cluster_dto.set_new_centroid(())
        self.assertRaises(ValueError, self.cluster_dto.set_new_centroid, ())

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_document_index_ideal(self):
        """
        Test adding a valid document index.
        """
        self.cluster_dto.add_document_index(1)
        self.assertIn(1, self.cluster_dto.get_indices())

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_document_index_duplicate(self):
        """
        Test adding a duplicate document index.
        """
        self.cluster_dto.add_document_index(1)
        self.cluster_dto.add_document_index(1)
        self.assertEqual(len(self.cluster_dto.get_indices()), 1)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_document_index_invalid_type(self):
        """
        Test adding a document index with invalid type.
        """
        self.assertRaises(ValueError, self.cluster_dto.add_document_index, "not_an_int")

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_document_index_negative(self):
        """
        Test adding a negative document index.
        """
        with self.assertRaises(ValueError):
            self.cluster_dto.add_document_index(-1)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_erase_indices_ideal(self):
        """
        Test erasing indices.
        """
        self.cluster_dto.add_document_index(1)
        self.cluster_dto.add_document_index(2)
        self.cluster_dto.erase_indices()
        self.assertEqual(len(self.cluster_dto.get_indices()), 0)

    @pytest.mark.lab_4_retrieval_w_clustering
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_len_method_ideal(self):
        """
        Test the __len__ method.
        """
        self.cluster_dto.add_document_index(1)
        self.cluster_dto.add_document_index(2)
        self.assertEqual(len(self.cluster_dto), 2)
