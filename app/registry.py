"""Lists model files in MODELS_DIR and lazy-loads them on first use."""
import logging
import os
from pathlib import Path
from threading import Lock

from app.predictor import SklearnPredictor

log = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))

_cache: dict[str, object] = {}
_cache_lock = Lock()
_listing_cache: list[dict] | None = None


def _humanize(stem):
    s = stem.replace("_", " ").replace("-", " ").strip()
    for suf in (" pipeline", " model"):
        if s.lower().endswith(suf):
            s = s[: -len(suf)]
    return s.title() or stem


def _discover():
    found = []
    seen = set()
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.iterdir()):
            if not p.is_file() or p.suffix not in (".pkl", ".joblib"):
                continue
            mid = p.stem
            if mid in seen:
                mid = f"{p.stem}{p.suffix}"
            seen.add(mid)
            found.append({"id": mid, "label": _humanize(p.stem), "path": str(p)})
    return found


def listing():
    global _listing_cache
    if _listing_cache is None:
        _listing_cache = _discover()
        log.info("registry: %d models under %s", len(_listing_cache), MODELS_DIR)
    return _listing_cache


def default_id():
    env_path = os.getenv("MODEL_PATH")
    items = listing()
    if env_path:
        try:
            target = Path(env_path).resolve()
            for it in items:
                if Path(it["path"]).resolve() == target:
                    return it["id"]
        except OSError:
            pass
    if items:
        return items[0]["id"]
    raise FileNotFoundError(f"No models in {MODELS_DIR}. Run scripts/setup_demo_models.sh.")


def get(model_id=None):
    if not model_id:
        model_id = default_id()
    with _cache_lock:
        if model_id in _cache:
            return _cache[model_id]
        entry = next((e for e in listing() if e["id"] == model_id), None)
        if entry is None:
            raise ValueError(f"unknown model_id: {model_id!r}")
        pred = SklearnPredictor(entry["path"])
        _cache[model_id] = pred
        return pred
