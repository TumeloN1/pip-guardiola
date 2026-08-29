"""Repo-relative paths used by ingest, features, training, and the API."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"

PLAYER_SEASONS_PARQUET = ARTIFACTS_DIR / "player_seasons.parquet"
FEATURES_PARQUET = ARTIFACTS_DIR / "features.parquet"
FEATURE_META_JSON = ARTIFACTS_DIR / "feature_meta.json"
EVAL_JSON = ARTIFACTS_DIR / "eval.json"
ENCODER_NPZ = ARTIFACTS_DIR / "encoder.npz"
EMBEDDINGS_NPZ = ARTIFACTS_DIR / "embeddings.npz"
GMM_NPZ = ARTIFACTS_DIR / "gmm.npz"
PROJECTION_NPZ = ARTIFACTS_DIR / "projection.npz"
