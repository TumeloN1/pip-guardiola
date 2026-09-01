"""Named archetype GMM: catalog assignment and keeper styles."""

from __future__ import annotations

import numpy as np

from kindred.cluster import (
    ARCHETYPE_CATALOG,
    N_ARCHETYPES,
    fit_gmm,
    keeper_archetypes,
    load_gmm,
    save_gmm,
    top_archetypes,
)
from kindred.features import GK_FEATURES, OUTFIELD_FEATURES


def test_catalog_covers_core_styles():
    names = [spec["name"] for spec in ARCHETYPE_CATALOG]
    assert len(names) == N_ARCHETYPES == 16
    assert len(set(names)) == 16
    for needed in (
        "Deep-lying playmaker",
        "Destroyer",
        "Ball-playing centre-back",
        "Inside forward",
        "Target striker",
        "Overlapping full-back",
    ):
        assert needed in names
    for spec in ARCHETYPE_CATALOG:
        assert not any("_p90" in spec["name"] or spec["name"].islower() and "_" in spec["name"] for _ in [0])
        for feat in spec["high"] + spec["low"]:
            assert feat in OUTFIELD_FEATURES, feat


def test_fit_assigns_unique_catalog_names():
    rng = np.random.default_rng(0)
    proto_idx = {n: i for i, n in enumerate(OUTFIELD_FEATURES)}
    X = rng.normal(size=(400, len(OUTFIELD_FEATURES)))
    for i, spec in enumerate(ARCHETYPE_CATALOG):
        block = X[i * 20 : (i + 1) * 20]
        for name in spec["high"]:
            block[:, proto_idx[name]] += 2.2
        for name in spec["low"]:
            block[:, proto_idx[name]] -= 1.4
    fit = fit_gmm(X, seed=0)
    assert fit.n_components == 16
    assert len(set(fit.labels)) == 16
    assert set(fit.labels) <= {spec["name"] for spec in ARCHETYPE_CATALOG}


def test_kdb_is_a_creator():
    from kindred.similarity import build_index
    import pandas as pd
    from kindred.paths import FEATURES_PARQUET

    index = build_index(pd.read_parquet(FEATURES_PARQUET), "outfield")
    x = index.features[index.id_to_row()["e46012d4-2020"]]
    names = {row["name"] for row in top_archetypes(load_gmm(), x)}
    assert names & {"Wide creator", "False nine", "Deep-lying playmaker", "Box-to-box midfielder"}


def test_keeper_archetypes_are_named():
    x = np.zeros(len(GK_FEATURES))
    x[0] = 2.0
    rows = keeper_archetypes(x)
    assert rows
    assert all(" " in r["name"] or r["name"][0].isupper() for r in rows)
    assert all("_" not in r["name"] for r in rows)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(320, len(OUTFIELD_FEATURES)))
    fit = fit_gmm(X, seed=1)
    path = tmp_path / "gmm.npz"
    save_gmm(fit, path)
    loaded = load_gmm(path)
    assert loaded.labels == fit.labels
    np.testing.assert_allclose(loaded.means, fit.means, atol=1e-5)
