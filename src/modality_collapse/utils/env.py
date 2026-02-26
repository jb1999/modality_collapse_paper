"""Environment setup for the modality collapse project.

Import this module early (before any transformers/huggingface_hub imports)
to configure HF_HOME if not already set.
"""

import os

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
