"""Ingest tests: drift fixes, transfer aggregation, KDB 2019-20 fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindred.ingest import (
    SEASON_MAX,
    SEASON_MIN,
    coalesce_columns,
    parse_age,
    parse_positions,
    player_season_id,
    season_label,
)
from kindred.paths import PLAYER_SEASONS_PARQUET


def test_parse_age_integer_and_years_days():
    assert parse_age(28) == 28.0
    assert parse_age("28") == 28.0
    assert parse_age("32-337") == pytest.approx(32 + 337 / 365.25)
    assert np.isnan(parse_age(None))
    assert np.isnan(parse_age(""))


def test_parse_positions_multivalued():
    assert parse_positions("MF") == ["MF"]
    assert parse_positions("FW,MF") == ["FW", "MF"]
    assert parse_positions("MF,DF") == ["MF", "DF"]
    assert parse_positions("") == []


def test_coalesce_prog_prgp_pattern():
    df = pd.DataFrame({
        "Prog": [10.0, np.nan, np.nan],
        "PrgP": [np.nan, np.nan, 20.0],
    })
    coalesce_columns(df, "Prog", "PrgP", out="pass_prg_p")
    assert df["pass_prg_p"].tolist() == pytest.approx([10.0, np.nan, 20.0], nan_ok=True)


def test_coalesce_xa_names():
    df = pd.DataFrame({
        "xA": [1.1, np.nan, np.nan],
        "xA_Expected": [np.nan, np.nan, 2.2],
    })
    coalesce_columns(df, "xA", "xA_Expected", out="xa")
    assert df["xa"].tolist() == pytest.approx([1.1, np.nan, 2.2], nan_ok=True)


def test_season_label_and_id():
    assert season_label(2020) == "2019-20"
    url = "https://fbref.com/en/players/e46012d4/Kevin-De-Bruyne"
    assert player_season_id(url, 2020) == "e46012d4-2020"


@pytest.fixture(scope="session")
def seasons() -> pd.DataFrame:
    if not PLAYER_SEASONS_PARQUET.exists():
        pytest.skip("player_seasons.parquet not built — run kindred-ingest")
    return pd.read_parquet(PLAYER_SEASONS_PARQUET)


def test_kdb_2019_20_fixture(seasons: pd.DataFrame):
    """Canonical verification row from the FBref standard table."""
    row = seasons.loc[seasons["player_id"] == "e46012d4-2020"]
    assert len(row) == 1
    kdb = row.iloc[0]
    assert kdb["player"] == "Kevin De Bruyne"
    assert kdb["squad"] == "Manchester City"
    assert kdb["pos"] == "MF"
    assert kdb["minutes"] == pytest.approx(2791)
    assert kdb["gls"] == pytest.approx(13)
    assert kdb["ast"] == pytest.approx(20)
    assert kdb["xg"] == pytest.approx(7.3)
    assert kdb["xag"] == pytest.approx(20.0)
    assert kdb["prg_p"] == pytest.approx(280)
    assert kdb["prg_c"] == pytest.approx(139)
    assert kdb["prg_r"] == pytest.approx(257)
    assert kdb["age"] == pytest.approx(28)


def test_player_season_unique_after_transfer_agg(seasons: pd.DataFrame):
    dupes = seasons.duplicated(["url", "season_end_year"]).sum()
    assert dupes == 0


def test_alexis_sanchez_2017_18_aggregated(seasons: pd.DataFrame):
    row = seasons.loc[
        (seasons["player"].eq("Alexis Sánchez"))
        & (seasons["season_end_year"].eq(2018))
        & (seasons["comp"].eq("Premier League"))
    ]
    assert len(row) == 1
    s = row.iloc[0]
    assert s["n_spells"] == 2
    assert "Arsenal" in s["squads"]
    assert "Manchester Utd" in s["squads"]
    # 1503 + 1043 minutes, 7+2 goals, 3+3 assists
    assert s["minutes"] == pytest.approx(2546)
    assert s["gls"] == pytest.approx(9)
    assert s["ast"] == pytest.approx(6)


def test_drift_fixes_hold_per_season(seasons: pd.DataFrame):
    window = seasons.loc[seasons["season_end_year"].between(SEASON_MIN, SEASON_MAX)]
    for year, group in window.groupby("season_end_year"):
        prg_null = group["prg_p"].isna().mean()
        xa_null = group["xa"].isna().mean()
        xag_null = group["xag"].isna().mean()
        pass_prg_null = group["pass_prg_p"].isna().mean()
        # Standard-table backbone should be populated; a few rows (no advanced
        # stats, typically 2024-25 gaps) may still be null.
        assert prg_null < 0.08, f"{year} prg_p null rate {prg_null:.2%}"
        assert xag_null < 0.08, f"{year} xag null rate {xag_null:.2%}"
        # Coalesced passing columns must not be 100% null in any season.
        assert xa_null < 0.15, f"{year} xa null rate {xa_null:.2%} — coalesce failed"
        assert pass_prg_null < 0.15, f"{year} pass_prg_p null rate {pass_prg_null:.2%}"
        assert pd.api.types.is_numeric_dtype(group["age"])
        # 2023-24 uses years-days; after parse, values are still ~15–42.
        ages = group["age"].dropna()
        assert ages.min() >= 14
        assert ages.max() < 50


def test_age_years_days_season_is_fractional(seasons: pd.DataFrame):
    y2024 = seasons.loc[seasons["season_end_year"].eq(2024), "age"].dropna()
    # Integer ages would all be whole numbers; years-days produces fractions.
    fractional = (y2024 % 1 != 0).mean()
    assert fractional > 0.5


def test_window_excludes_2026(seasons: pd.DataFrame):
    years = set(seasons["season_end_year"].unique())
    assert years <= set(range(SEASON_MIN, SEASON_MAX + 1))
    assert 2018 in years and 2025 in years
