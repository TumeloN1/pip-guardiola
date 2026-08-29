"""Feature-engineering unit tests (rates, shares, possession-adjust, z-score)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindred.features import (
    DEFAULT_MIN_MINUTES,
    FEATURE_GROUPS,
    OUTFIELD_FEATURES,
    per90,
    possession_adjust,
    share,
    zscore_within,
)
from kindred.paths import FEATURES_PARQUET


def test_per90_and_zero_minutes():
    minutes = pd.Series([900.0, 0.0, 180.0])
    goals = pd.Series([10.0, 1.0, 2.0])
    rates = per90(goals, minutes)
    assert rates.iloc[0] == pytest.approx(1.0)
    assert np.isnan(rates.iloc[1])
    assert rates.iloc[2] == pytest.approx(1.0)


def test_share_zero_denominator_is_zero_not_null():
    part = pd.Series([5.0, 0.0, 2.0])
    whole = pd.Series([10.0, 0.0, 0.0])
    out = share(part, whole)
    assert out.iloc[0] == pytest.approx(0.5)
    assert out.iloc[1] == pytest.approx(0.0)
    assert out.iloc[2] == pytest.approx(0.0)


def test_possession_adjust_city_vs_low_block():
    rate = pd.Series([2.0, 2.0])
    poss = pd.Series([66.2, 50.0])
    padj = possession_adjust(rate, poss)
    assert padj.iloc[1] == pytest.approx(2.0)
    assert padj.iloc[0] == pytest.approx(2.0 * 50.0 / (100.0 - 66.2))
    assert padj.iloc[0] > padj.iloc[1]


def test_zscore_within_cohort_is_mean_zero():
    df = pd.DataFrame({
        "season_end_year": [2020, 2020, 2020, 2021, 2021, 2021],
        "comp": ["Premier League"] * 6,
        "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    })
    z = zscore_within(df, ["x"], ["season_end_year", "comp"])
    for _, g in z.groupby(["season_end_year", "comp"]):
        assert g["x"].mean() == pytest.approx(0.0, abs=1e-9)


def test_feature_groups_cover_outfield_list():
    grouped = [f for cols in FEATURE_GROUPS.values() for f in cols]
    assert grouped == OUTFIELD_FEATURES
    assert len(OUTFIELD_FEATURES) >= 40


@pytest.fixture(scope="session")
def features() -> pd.DataFrame:
    if not FEATURES_PARQUET.exists():
        pytest.skip("features.parquet not built — run kindred-features")
    return pd.read_parquet(FEATURES_PARQUET)


def test_kdb_survives_minutes_filter(features: pd.DataFrame):
    row = features.loc[features["player_id"] == "e46012d4-2020"]
    assert len(row) == 1
    kdb = row.iloc[0]
    assert kdb["role"] == "outfield"
    assert kdb["minutes"] >= DEFAULT_MIN_MINUTES
    assert kdb["z_xag_p90"] > 2.0
    assert kdb["z_kp_p90"] > 2.0


def test_zscored_columns_exist_and_finite(features: pd.DataFrame):
    outfield = features.loc[features["role"] == "outfield"]
    for col in [f"z_{name}" for name in OUTFIELD_FEATURES]:
        assert col in outfield.columns
        assert np.isfinite(outfield[col]).mean() > 0.99
