"""Rates, style shares, possession-adjusted defending, and within-era z-scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kindred.paths import (
    ARTIFACTS_DIR,
    FEATURE_META_JSON,
    FEATURES_PARQUET,
    PLAYER_SEASONS_PARQUET,
)

DEFAULT_MIN_MINUTES = 900.0

OUTFIELD_RATE_COUNTS = {
    "npxg_p90": "npxg",
    "xag_p90": "xag",
    "xa_p90": "xa",
    "gls_p90": "gls",
    "ast_p90": "ast",
    "sh_p90": "sh",
    "sot_p90": "sot",
    "kp_p90": "kp",
    "prg_p_p90": "prg_p",
    "prg_c_p90": "prg_c",
    "prg_r_p90": "prg_r",
    "ppa_p90": "ppa",
    "tb_p90": "tb",
    "crosses_p90": "crs_pass",
    "pass_att_p90": "pass_att",
    "pass_final_third_p90": "pass_final_third",
    "touches_p90": "touches",
    "take_att_p90": "take_att",
    "carries_p90": "carries",
    "cpa_p90": "cpa",
    "tkl_p90": "tkl",
    "int_p90": "interceptions",
    "blocks_p90": "blocks",
    "clr_p90": "clearances",
    "recov_p90": "recoveries",
    "aerial_p90": "aerial_won",
    "fls_p90": "fls",
    "dis_p90": "dispossessed",
}

PADJ_RATES = {
    "padj_tkl": "tkl_p90",
    "padj_int": "int_p90",
    "padj_blocks": "blocks_p90",
    "padj_clr": "clr_p90",
    "padj_recov": "recov_p90",
    "padj_aerial": "aerial_p90",
}

FEATURE_GROUPS: dict[str, list[str]] = {
    "finishing": ["npxg_p90", "gls_p90", "sh_p90", "sot_p90", "shot_dist"],
    "creation": ["xag_p90", "xa_p90", "kp_p90", "ppa_p90", "tb_p90", "ast_p90"],
    "passing": [
        "prg_p_p90",
        "pass_att_p90",
        "prg_pass_share",
        "pass_short_share",
        "pass_med_share",
        "pass_long_share",
        "pass_final_third_p90",
    ],
    "carrying": [
        "prg_c_p90",
        "prg_r_p90",
        "take_att_p90",
        "takeons_per_touch",
        "dist_per_carry",
        "cpa_p90",
        "carries_p90",
    ],
    "occupation": [
        "touch_def_pen_share",
        "touch_def3_share",
        "touch_mid3_share",
        "touch_att3_share",
        "touch_att_pen_share",
    ],
    "defending": [
        "padj_tkl",
        "padj_int",
        "padj_blocks",
        "padj_clr",
        "padj_recov",
        "tkl_def_share",
        "tkl_mid_share",
        "tkl_att_share",
    ],
    "duels": ["aerial_p90", "aerial_win_share", "take_succ_share", "fls_p90", "dis_p90"],
}

GK_FEATURES = [
    "gk_save_share",
    "gk_cs_p90",
    "gk_psxg_plus_minus_p90",
    "gk_launch_pct",
    "gk_avg_pass_len",
    "gk_goal_kick_launch_pct",
    "gk_avg_goal_kick_len",
    "gk_cross_stop_share",
    "gk_opa_p90",
    "gk_sweeper_avg_dist",
    "gk_throws_share",
    "gk_launch_cmp_share",
    "gk_pka_save_share",
]

GK_GROUPS: dict[str, list[str]] = {
    "shotstopping": ["gk_save_share", "gk_psxg_plus_minus_p90", "gk_pka_save_share"],
    "distribution": [
        "gk_launch_pct",
        "gk_avg_pass_len",
        "gk_goal_kick_launch_pct",
        "gk_avg_goal_kick_len",
        "gk_throws_share",
        "gk_launch_cmp_share",
    ],
    "sweeping": ["gk_opa_p90", "gk_sweeper_avg_dist", "gk_cross_stop_share", "gk_cs_p90"],
}

OUTFIELD_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
assert len(OUTFIELD_FEATURES) == len(set(OUTFIELD_FEATURES))
assert len(OUTFIELD_FEATURES) >= 40


def per90(count: pd.Series, minutes: pd.Series) -> pd.Series:
    minutes = pd.to_numeric(minutes, errors="coerce")
    count = pd.to_numeric(count, errors="coerce")
    out = pd.Series(np.nan, index=count.index, dtype=float)
    ok = minutes > 0
    out.loc[ok] = count.loc[ok] * 90.0 / minutes.loc[ok]
    return out


def share(part: pd.Series, whole: pd.Series) -> pd.Series:
    part = pd.to_numeric(part, errors="coerce").fillna(0.0)
    whole = pd.to_numeric(whole, errors="coerce")
    out = pd.Series(np.nan, index=part.index, dtype=float)
    ok = whole > 0
    out.loc[ok] = part.loc[ok] / whole.loc[ok]
    out.loc[whole.fillna(0) == 0] = 0.0
    return out


def possession_adjust(rate_per90: pd.Series, team_poss: pd.Series) -> pd.Series:
    """padj = raw_per90 * 50 / (100 - team_possession).

    City defenders look passive because they hold the ball; this puts them on
    the same scale as a side that defends 60 minutes a night.
    """
    poss = pd.to_numeric(team_poss, errors="coerce")
    rate = pd.to_numeric(rate_per90, errors="coerce")
    denom = 100.0 - poss
    out = pd.Series(np.nan, index=rate.index, dtype=float)
    ok = denom.gt(5) & denom.notna() & rate.notna()
    out.loc[ok] = rate.loc[ok] * 50.0 / denom.loc[ok]
    return out


def zscore_within(frame: pd.DataFrame, columns: list[str], keys: list[str]) -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby(keys, dropna=False)
    for col in columns:
        mu = grouped[col].transform("mean")
        sd = grouped[col].transform("std")
        z = (out[col] - mu) / sd.replace(0, np.nan)
        out[col] = z.replace([np.inf, -np.inf], np.nan)
    return out


def _outfield_style(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    minutes = out["minutes"]
    for feat, src in OUTFIELD_RATE_COUNTS.items():
        out[feat] = per90(out[src], minutes)

    touches = out["touches"]
    out["touch_def_pen_share"] = share(out["touches_def_pen"], touches)
    out["touch_def3_share"] = share(out["touches_def3"], touches)
    out["touch_mid3_share"] = share(out["touches_mid3"], touches)
    out["touch_att3_share"] = share(out["touches_att3"], touches)
    out["touch_att_pen_share"] = share(out["touches_att_pen"], touches)

    tkl = out["tkl"]
    out["tkl_def_share"] = share(out["tkl_def3"], tkl)
    out["tkl_mid_share"] = share(out["tkl_mid3"], tkl)
    out["tkl_att_share"] = share(out["tkl_att3"], tkl)

    pass_att = out["pass_att"]
    out["pass_short_share"] = share(out["pass_att_short"], pass_att)
    out["pass_med_share"] = share(out["pass_att_med"], pass_att)
    out["pass_long_share"] = share(out["pass_att_long"], pass_att)
    out["prg_pass_share"] = share(out["prg_p"], pass_att)

    carries = out["carries"]
    out["dist_per_carry"] = share(out["carry_totdist"], carries)
    out["takeons_per_touch"] = share(out["take_att"], touches)
    out["take_succ_share"] = share(out["take_succ"], out["take_att"])
    aerials = pd.to_numeric(out["aerial_won"], errors="coerce").fillna(0) + pd.to_numeric(
        out["aerial_lost"], errors="coerce"
    ).fillna(0)
    out["aerial_win_share"] = share(out["aerial_won"], aerials.replace(0, np.nan).fillna(0))

    # Average shot distance is already an average; trust it only when the
    # player actually shot. Zero-shot rows stay NaN and are median-filled
    # inside the z-score cohort so they don't dominate finishing.
    out["shot_dist"] = pd.to_numeric(out["shot_dist"], errors="coerce")
    out.loc[pd.to_numeric(out["sh"], errors="coerce").fillna(0) <= 0, "shot_dist"] = np.nan

    for dest, src in PADJ_RATES.items():
        out[dest] = possession_adjust(out[src], out["team_poss"])
    return out


def _gk_style(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    minutes = out["minutes"]
    out["gk_save_share"] = share(out["gk_saves"], out["gk_sota"])
    out["gk_cs_p90"] = per90(out["gk_cs"], minutes)
    out["gk_psxg_plus_minus_p90"] = per90(out["gk_psxg_plus_minus"], minutes)
    out["gk_cross_stop_share"] = share(out["gk_crosses_stopped"], out["gk_crosses_faced"])
    out["gk_opa_p90"] = per90(out["gk_opa"], minutes)
    out["gk_throws_share"] = share(out["gk_throws"], out["gk_pass_att"])
    out["gk_launch_cmp_share"] = share(out["gk_launch_cmp"], out["gk_launch_att"])
    out["gk_pka_save_share"] = share(out["gk_pksv"], out["gk_pkatt"])
    # Keep the already-average keeper columns; they are not ratio-of-zeros traps
    # in the same way as SoT%. They still get z-scored.
    for col in [
        "gk_launch_pct",
        "gk_avg_pass_len",
        "gk_goal_kick_launch_pct",
        "gk_avg_goal_kick_len",
        "gk_sweeper_avg_dist",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _median_fill_within(frame: pd.DataFrame, columns: list[str], keys: list[str]) -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby(keys, dropna=False)
    for col in columns:
        med = grouped[col].transform("median")
        global_med = out[col].median()
        filled = out[col].fillna(med).fillna(global_med)
        out[col] = filled
    return out


def build_features(
    seasons: pd.DataFrame,
    *,
    min_minutes: float = DEFAULT_MIN_MINUTES,
) -> tuple[pd.DataFrame, dict]:
    eligible = seasons.loc[seasons["minutes"].fillna(0) >= min_minutes].copy()

    outfield_src = eligible.loc[~eligible["is_gk"]].copy()
    gk_src = eligible.loc[eligible["is_gk"]].copy()

    outfield = _outfield_style(outfield_src)
    required_outfield = ["touches", "pass_att", "tkl", "team_poss"]
    outfield = outfield.dropna(subset=required_outfield)

    gk = _gk_style(gk_src)
    gk = gk.dropna(subset=["gk_sota", "gk_saves"])

    cohort_keys = ["season_end_year", "comp"]
    outfield = _median_fill_within(outfield, OUTFIELD_FEATURES, cohort_keys)
    gk = _median_fill_within(gk, GK_FEATURES, cohort_keys)

    outfield_raw = outfield.copy()
    gk_raw = gk.copy()

    outfield_z = zscore_within(outfield, OUTFIELD_FEATURES, cohort_keys)
    gk_z = zscore_within(gk, GK_FEATURES, cohort_keys)

    meta_cols = [
        "player_id",
        "fbref_id",
        "url",
        "player",
        "season",
        "season_end_year",
        "squad",
        "squads",
        "n_spells",
        "comp",
        "pos",
        "primary_pos",
        "is_gk",
        "age",
        "nation",
        "minutes",
        "team_poss",
    ]

    def _pack(zdf: pd.DataFrame, raw: pd.DataFrame, feats: list[str], role: str) -> pd.DataFrame:
        zpart = zdf[meta_cols + feats].copy()
        zpart = zpart.rename(columns={c: f"z_{c}" for c in feats})
        raw_part = raw[feats].copy()
        raw_part = raw_part.rename(columns={c: f"raw_{c}" for c in feats})
        packed = pd.concat([zpart.reset_index(drop=True), raw_part.reset_index(drop=True)], axis=1)
        packed["role"] = role
        return packed

    out_pack = _pack(outfield_z, outfield_raw, OUTFIELD_FEATURES, "outfield")
    gk_pack = _pack(gk_z, gk_raw, GK_FEATURES, "keeper")
    packed = pd.concat([out_pack, gk_pack], ignore_index=True, sort=False)

    # Poisson-augmentation counts live on the original eligible rows.
    poisson_cols = [
        "npxg",
        "xag",
        "xa",
        "gls",
        "ast",
        "sh",
        "sot",
        "kp",
        "prg_p",
        "prg_c",
        "prg_r",
        "ppa",
        "tb",
        "crs_pass",
        "pass_att",
        "pass_att_short",
        "pass_att_med",
        "pass_att_long",
        "pass_final_third",
        "touches",
        "touches_def_pen",
        "touches_def3",
        "touches_mid3",
        "touches_att3",
        "touches_att_pen",
        "take_att",
        "take_succ",
        "carries",
        "carry_totdist",
        "cpa",
        "tkl",
        "tkl_def3",
        "tkl_mid3",
        "tkl_att3",
        "interceptions",
        "blocks",
        "clearances",
        "recoveries",
        "aerial_won",
        "aerial_lost",
        "fls",
        "dispossessed",
        "gk_saves",
        "gk_sota",
        "gk_cs",
        "gk_psxg_plus_minus",
        "gk_crosses_stopped",
        "gk_crosses_faced",
        "gk_opa",
        "gk_throws",
        "gk_pass_att",
        "gk_launch_cmp",
        "gk_launch_att",
        "gk_pksv",
        "gk_pkatt",
    ]
    poisson_present = [c for c in poisson_cols if c in eligible.columns]
    poisson = eligible[["player_id", *poisson_present]].copy()
    packed = packed.merge(poisson, on="player_id", how="left")

    z_params: dict = {"outfield": {}, "keeper": {}}
    for role, src, feats in (
        ("outfield", outfield_raw, OUTFIELD_FEATURES),
        ("keeper", gk_raw, GK_FEATURES),
    ):
        for (year, comp), g in src.groupby(cohort_keys, dropna=False):
            key = f"{int(year)}|{comp}"
            z_params[role][key] = {
                feat: {
                    "mean": float(g[feat].mean()) if pd.notna(g[feat].mean()) else 0.0,
                    "std": float(g[feat].std()) if pd.notna(g[feat].std()) and g[feat].std() else 1.0,
                }
                for feat in feats
            }

    meta = {
        "min_minutes": min_minutes,
        "outfield_features": OUTFIELD_FEATURES,
        "gk_features": GK_FEATURES,
        "groups": FEATURE_GROUPS,
        "gk_groups": GK_GROUPS,
        "cohort_keys": cohort_keys,
        "z_params": z_params,
        "n_outfield": int((packed["role"] == "outfield").sum()),
        "n_keeper": int((packed["role"] == "keeper").sum()),
    }
    return packed, meta


def run(
    *,
    source: Path | None = None,
    output: Path | None = None,
    meta_output: Path | None = None,
    min_minutes: float = DEFAULT_MIN_MINUTES,
) -> Path:
    src = source or PLAYER_SEASONS_PARQUET
    if not src.exists():
        raise FileNotFoundError(f"{src} missing — run kindred-ingest first")
    seasons = pd.read_parquet(src)
    packed, meta = build_features(seasons, min_minutes=min_minutes)
    dest = output or FEATURES_PARQUET
    dest.parent.mkdir(parents=True, exist_ok=True)
    packed.to_parquet(dest, index=False)
    meta_path = meta_output or FEATURE_META_JSON
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"wrote {dest}  outfield={meta['n_outfield']:,}  keepers={meta['n_keeper']:,}  "
        f"outfield_dims={len(OUTFIELD_FEATURES)}  gk_dims={len(GK_FEATURES)}"
    )
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Kindred feature matrix")
    parser.add_argument("--min-minutes", type=float, default=DEFAULT_MIN_MINUTES)
    args = parser.parse_args(argv)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    run(min_minutes=args.min_minutes)


if __name__ == "__main__":
    main()
