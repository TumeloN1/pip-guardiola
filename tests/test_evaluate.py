"""Smoke tests for the baseline retrieval metric."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindred.paths import FEATURES_PARQUET
from kindred.similarity import build_index, similar


@pytest.fixture(scope="session")
def features() -> pd.DataFrame:
    if not FEATURES_PARQUET.exists():
        pytest.skip("features.parquet not built")
    return pd.read_parquet(FEATURES_PARQUET)


def test_kdb_neighbours_are_attacking_mids(features: pd.DataFrame):
    index = build_index(features, role="outfield")
    hits = similar(index, "e46012d4-2020", k=10, comps=["Premier League"])
    assert hits
    assert hits[0]["player"] != "Kevin De Bruyne" or hits[0]["season"] != "2019-20"
    pos = {h["pos"] for h in hits}
    assert any("MF" in p for p in pos)


def test_pca_whiten_shapes(features: pd.DataFrame):
    from kindred.similarity import pca_whiten

    index = build_index(features, role="outfield")
    white = pca_whiten(index.features, n_components=16)
    assert white.shape == (index.features.shape[0], 16)
    # Whitened features should be roughly unit-variance.
    assert np.std(white[:, 0]) == pytest.approx(1.0, rel=0.15)
