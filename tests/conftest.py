"""Point the registry at the local models/ before app.main imports it."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MODELS_DIR", str(REPO / "models"))
