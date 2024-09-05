"""
Checks the first lab text preprocessing functions
"""

import unittest

import pytest

from lab_1.main import say_hello_world


class ExampleTest(unittest.TestCase):
    """
    Tests tokenize function
    """

    @pytest.mark.lab_1
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_ideal(self):
        """
        Ideal tokenize scenario
        """
        say_hello_world()
        self.assertEqual(10, 10)
