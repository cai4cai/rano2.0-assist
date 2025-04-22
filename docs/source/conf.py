# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path('..', '..').resolve()))  # So Sphinx can find your module
sys.path.insert(0, str(Path('..', '..', 'RANO').resolve()))

project = 'RANO2.0-assist'
copyright = '2025, Aaron Kujawa'
author = 'Aaron Kujawa'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'autoapi.extension',
    'sphinx.ext.autosectionlabel',   # for auto-generating section labels
    "sphinx.ext.napoleon", # Google/NumPy-style docstrings
    'sphinx.ext.viewcode',  # add links to source code
]


autoapi_dirs = ['/home/slicer/Projects/rano2.0-assist/RANO', ]

# autosummary_generate = True  # Generate API documentation automatically
# autosummary_imported_members = True  # Include members of imported modules

myst_enable_extensions = ["colon_fence"]  # For markdown support

source_suffix = ['.md', '.rst']  # Support both markdown and reStructuredText because autoapi creates .rst files

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
