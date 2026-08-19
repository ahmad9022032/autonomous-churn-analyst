"""Compatibility shim — the Streamlit app lives in streamlit-app/streamlit_app.py.

Streamlit Community Cloud pins the deployed main-file path and does not allow
editing it, so this file keeps the live URL working after the repo was
reorganized into streamlit-app/ and react-app/.
"""

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "streamlit-app" / "streamlit_app.py"),
    run_name="__main__",
)
