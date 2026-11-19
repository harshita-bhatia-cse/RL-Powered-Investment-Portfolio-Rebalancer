"""HTML dashboard application entry point.

The analytics API owns the FastAPI application.  This small module provides a
stable Uvicorn import path without coupling the dashboard to RL or trading code.
"""

import sys
from pathlib import Path

# Direct execution (`python src/main.py dashboard`) places ``src/`` rather than
# the repository root on sys.path.  Add the root so every backend module keeps
# its canonical ``src.*`` import path.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.api.analytics_api import app

