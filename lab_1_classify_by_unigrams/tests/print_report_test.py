"""
Checks report printing for detection of language
"""

import unittest

import pytest

from lab_1_classify_by_unigrams.main import print_report


class PrintReportTest(unittest.TestCase):
    """
    Tests report printing function
    """

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_print_report_ideal(self):
        """
        Ideal scenario
        """
        expected = None
        detections = [('es', 0.0016), ('tr', 0.0018), ('it', 0.002), ('fr', 0.0021),
                      ('en', 0.0022), ('de', 0.0024), ('ru', 0.0037)]

        self.assertEqual(expected, print_report(detections))

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_print_report_bad_input(self):
        """
        Bad input scenario
        """

        detections = {}

        expected = None

        self.assertEqual(expected, print_report(detections))
