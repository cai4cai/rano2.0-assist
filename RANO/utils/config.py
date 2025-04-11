import os

debug = False
module_path = os.path.dirname(os.path.dirname(__file__))  # resolves to the directory that contains RANO.py
dynunet_pipeline_path = os.path.join(module_path, "..", "dynunet_pipeline")
