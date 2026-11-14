"""Hosting entrypoint for the existing Streamlit interface."""

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# A fresh hosting server needs to download BGE before it can search.
os.environ["EDI_ALLOW_MODEL_DOWNLOADS"] = "1"

# Run the existing interface on every Streamlit rerun.
runpy.run_path(
    str(PROJECT_ROOT / "app_interface.py"),
    run_name="__main__",
)