"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
# pylint: disable=invalid-name,redefined-builtin

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

project = 'Лабораторный Практикум и Курс Лекций'
copyright = '2023, Демидовский А.В. и другие'
author = 'Демидовский А.В. и другие'

extensions = [
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

exclude_patterns = [
    'venv/*',
    'docs/private/*'
]

language = 'ru'

html_theme = 'sphinx_rtd_theme'
