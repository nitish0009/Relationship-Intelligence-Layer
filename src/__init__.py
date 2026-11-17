"""Relationship Intelligence Layer research prototype."""

# XGBoost must load its OpenMP runtime before PyTorch does: on macOS, if torch's libomp is
# loaded first, XGBoost's training segfaults (exit 139). Importing it here guarantees the
# order for every entry point, because the package __init__ runs before any src.* module.
try:  # pragma: no cover - environment dependent
    import xgboost  # noqa: F401
except Exception:
    pass
