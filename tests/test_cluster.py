"""Named archetype GMM: catalog assignment and keeper styles."""

from __future__ import annotations

import numpy as np

from kindred.cluster import (
    ARCHETYPE_CATALOG,
    DISPLAY_THRESHOLD,
    N_ARCHETYPES,
    fit_gmm,
    keeper_archetypes,
    leading_style_names,
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
        assert spec["positions"], spec["name"]
        assert set(spec["positions"]) <= {"DF", "MF", "FW"}
        assert not any("_p90" in spec["name"] or spec["name"].islower() and "_" in spec["name"] for _ in [0])
        for feat in spec["high"] + spec["low"]:
            assert feat in OUTFIELD_FEATURES, feat
    false_nine = next(s for s in ARCHETYPE_CATALOG if s["name"] == "False nine")
    assert false_nine["positions"] == ["FW"]
    overlapping = next(s for s in ARCHETYPE_CATALOG if s["name"] == "Overlapping full-back")
    assert overlapping["positions"] == ["DF"]


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


def _allowed_names(pos: str) -> set[str]:
    tokens = {p.strip() for p in pos.split(",") if p.strip()}
    return {spec["name"] for spec in ARCHETYPE_CATALOG if tokens.intersection(spec["positions"])}


def test_fullback_vector_cannot_be_false_nine():
    proto_idx = {n: i for i, n in enumerate(OUTFIELD_FEATURES)}
    x = np.zeros(len(OUTFIELD_FEATURES))
    false_nine = next(s for s in ARCHETYPE_CATALOG if s["name"] == "False nine")
    for name in false_nine["high"]:
        x[proto_idx[name]] = 2.4
    rows = top_archetypes(None, x, pos="DF", primary_pos="DF")
    names = {row["name"] for row in rows}
    assert "False nine" not in names
    assert "Poacher" not in names
    assert "Wide creator" not in names
    assert names <= _allowed_names("DF")
    assert all(row["weight"] >= DISPLAY_THRESHOLD for row in rows)


def test_hybrid_df_mf_keeps_both_families():
    proto_idx = {n: i for i, n in enumerate(OUTFIELD_FEATURES)}
    x = np.zeros(len(OUTFIELD_FEATURES))
    for name in ("prg_c_p90", "padj_tkl", "touch_att3_share", "xag_p90", "kp_p90"):
        x[proto_idx[name]] = 1.8
    rows = top_archetypes(None, x, pos="DF,MF", primary_pos="DF")
    names = {row["name"] for row in rows}
    assert names <= _allowed_names("DF,MF")
    assert "False nine" not in names
    assert names & {
        "Wing-back",
        "Overlapping full-back",
        "Box-to-box midfielder",
        "Destroyer",
        "Wide creator",
        "Inverted full-back",
    }


def test_kdb_is_a_creator():
    from kindred.similarity import build_index
    import pandas as pd
    from kindred.paths import FEATURES_PARQUET

    index = build_index(pd.read_parquet(FEATURES_PARQUET), "outfield")
    i = index.id_to_row()["e46012d4-2020"]
    rows = top_archetypes(
        load_gmm(),
        index.features[i],
        pos=str(index.positions[i]),
        primary_pos=str(index.primary_pos[i]),
    )
    names = {row["name"] for row in rows}
    assert names <= _allowed_names(str(index.positions[i]))
    assert names & {
        "Wide creator",
        "Deep-lying playmaker",
        "Box-to-box midfielder",
        "Shadow striker",
    }
    assert "False nine" not in names
    assert all(row["weight"] >= DISPLAY_THRESHOLD for row in rows)


def test_taa_is_not_a_false_nine():
    from kindred.similarity import build_index
    import pandas as pd
    from kindred.paths import FEATURES_PARQUET

    index = build_index(pd.read_parquet(FEATURES_PARQUET), "outfield")
    i = index.id_to_row()["cd1acf9d-2020"]
    rows = top_archetypes(
        load_gmm(),
        index.features[i],
        pos=str(index.positions[i]),
        primary_pos=str(index.primary_pos[i]),
    )
    names = {row["name"] for row in rows}
    assert str(index.positions[i]) == "DF"
    assert names <= _allowed_names("DF")
    assert "False nine" not in names
    assert "Poacher" not in names
    assert "Wide creator" not in names
    assert names & {
        "Overlapping full-back",
        "Inverted full-back",
        "Wing-back",
        "Ball-playing centre-back",
        "Destroyer",
    }
    assert all(row["weight"] >= DISPLAY_THRESHOLD for row in rows)


def test_leading_style_names_match_position_gate():
    from kindred.similarity import build_index
    import pandas as pd
    from kindred.paths import FEATURES_PARQUET

    index = build_index(pd.read_parquet(FEATURES_PARQUET), "outfield")
    names = leading_style_names(index.features, index.positions, index.primary_pos)
    taa = names[index.id_to_row()["cd1acf9d-2020"]]
    kdb = names[index.id_to_row()["e46012d4-2020"]]
    assert taa in _allowed_names("DF")
    assert taa not in {"False nine", "Poacher", "Wide creator"}
    assert kdb in _allowed_names("MF")
    assert kdb != "False nine"


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
