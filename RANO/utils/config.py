"""
This module contains configuration settings for the RANO module.
"""
import os

debug = False
"""Set to True to enable debug mode."""

module_path = os.path.dirname(os.path.dirname(__file__))
"""Path to the module directory. Resolves to the directory that contains RANO.py."""

dynunet_pipeline_path = os.path.join(module_path, "..", "dynunet_pipeline")
"""Path to the dynunet_pipeline directory used for running the segmentation pipeline."""
