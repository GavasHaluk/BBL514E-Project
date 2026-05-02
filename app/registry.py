"""Lists model files in MODELS_DIR and lazy-loads them on first use.

The directory listing is cached and invalidated whenever MODELS_DIR's mtime
changes, so dropping a new joblib in `models/` is picked up on the next
/api/models hit -- no container restart needed.
"""
import logging
import os
from pathlib import Path
from threading import Lock

from app.predictor import SklearnPredictor

log = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
ALLOWED_SUFFIXES = (".pkl", ".joblib")

_cache: dict[str, object] = {}
_cache_lock = Lock()
_listing: list[dict] | None = None
_listing_fp: tuple | None = None


def _humanize(stem):
    s = stem.replace("_", " ").replace("-", " ").strip()
    for suf in (" pipeline", " model"):
        if s.lower().endswith(suf):
            s = s[: -len(suf)]
    return s.title() or stem


def _dir_fingerprint():
    """Hash the per-file (name, mtime, size) tuples. Survives the macOS
    Docker quirk where directory mtime doesn't propagate through bind
    mounts -- file stats do."""
    if not MODELS_DIR.exists():
        return ()
    entries = []
    for p in MODELS_DIR.iterdir():
        if p.is_file() and p.suffix in ALLOWED_SUFFIXES:
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((p.name, st.st_mtime, st.st_size))
    return tuple(sorted(entries))


def _discover():
    found = []
    seen = set()
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.iterdir()):
            if not p.is_file() or p.suffix not in ALLOWED_SUFFIXES:
                continue
            mid = p.stem
            if mid in seen:
                # collision between e.g. foo.pkl and foo.joblib -- disambiguate
                mid = f"{p.stem}{p.suffix}"
            seen.add(mid)
            found.append({"id": mid, "label": _humanize(p.stem), "path": str(p)})
    return found


def listing(force=False):
    """Return the discovered models, refreshing if MODELS_DIR has changed."""
    global _listing, _listing_fp
    fp = _dir_fingerprint()
    if force or _listing is None or fp != _listing_fp:
        _listing = _discover()
        _listing_fp = fp
        # Drop cached predictors whose file is gone, so a delete-then-restore
        # cycle doesn't keep serving the stale handle.
        with _cache_lock:
            live_ids = {it["id"] for it in _listing}
            for stale in [k for k in _cache if k not in live_ids]:
                _cache.pop(stale, None)
        log.info("registry: %d models under %s", len(_listing), MODELS_DIR)
    return _listing


def default_id():
    items = listing()
    if not items:
        raise FileNotFoundError(f"No models in {MODELS_DIR}.")
    return items[0]["id"]


def get(model_id=None):
    if not model_id:
        model_id = default_id()
    with _cache_lock:
        if model_id in _cache:
            return _cache[model_id]
    # Refresh outside the lock so a slow load doesn't block the listing.
    entry = next((e for e in listing() if e["id"] == model_id), None)
    if entry is None:
        raise ValueError(f"unknown model_id: {model_id!r}")
    pred = SklearnPredictor(entry["path"])
    with _cache_lock:
        _cache.setdefault(model_id, pred)
        return _cache[model_id]


def describe(model_id):
    """Lightweight per-model metadata for /api/models. Loads the artifact
    on first call (so this is not free), but the result is cached."""
    pred = get(model_id)
    model = pred.model
    classes = getattr(model, "classes_", None)
    return {
        "n_features_in": int(getattr(model, "n_features_in_", 0)) or None,
        "feature_names_in": pred.feature_names,
        "classes": [c.item() if hasattr(c, "item") else c for c in classes] if classes is not None else None,
        "version": pred.version,
        "trained_at": pred.trained_at,
    }
