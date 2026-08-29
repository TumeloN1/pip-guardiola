"""Download FBref Big 5 release RDS files, repair schema drift, aggregate transfers.

Data comes from JaseZiv/worldfootballR_data GitHub *release assets* under the
tag ``fb_big5_advanced_season_stats`` — not the stale in-repo ``data/`` folder
of that project, which stops mid 2022-23.

Usable window is Season_End_Year 2018–2025 (2017-18 through 2024-25): that is
the overlap of the advanced passing / possession / defense tables. 2025-26 is
present in standard/shooting/passing but missing possession and defense, so it
is excluded until the mirror catches up.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

from kindred.paths import ARTIFACTS_DIR, PLAYER_SEASONS_PARQUET, RAW_DIR

RELEASE_TAG = "fb_big5_advanced_season_stats"
RELEASE_BASE = (
    "https://github.com/JaseZiv/worldfootballR_data/releases/download/"
    f"{RELEASE_TAG}"
)

PLAYER_TABLES = (
    "standard",
    "shooting",
    "passing",
    "passing_types",
    "possession",
    "defense",
    "misc",
    "playing_time",
    "keepers",
    "keepers_adv",
)
TEAM_TABLES = ("possession",)

SEASON_MIN = 2018
SEASON_MAX = 2025

JOIN_KEYS = ["Url", "Season_End_Year", "Squad"]
PLAYER_SEASON_KEYS = ["Url", "Season_End_Year"]

# FBref player URLs look like https://fbref.com/en/players/<id>/Name
_FBREF_ID_RE = re.compile(r"/players/([a-f0-9]+)/", re.I)


def parse_age(value) -> float:
    """Parse FBref Age: an integer year, or ``'32-337'`` (years-days) from 2023-24."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, float):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return np.nan
    if "-" in text:
        years, days = text.split("-", 1)
        try:
            return float(years) + float(days) / 365.25
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_positions(pos: object) -> list[str]:
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return []
    text = str(pos).strip()
    if not text:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def primary_position(pos: object) -> str:
    parsed = parse_positions(pos)
    return parsed[0] if parsed else ""


def fbref_player_id(url: object) -> str:
    if not isinstance(url, str):
        return ""
    match = _FBREF_ID_RE.search(url)
    return match.group(1) if match else ""


def player_season_id(url: object, season_end_year: object) -> str:
    return f"{fbref_player_id(url)}-{int(season_end_year)}"


def season_label(season_end_year: int) -> str:
    start = int(season_end_year) - 1
    return f"{start}-{str(int(season_end_year))[-2:]}"


def coalesce_columns(df: pd.DataFrame, *names: str, out: str) -> pd.DataFrame:
    """First non-null among ``names`` → ``out``. Used for Prog/PrgP, xA/xA_Expected, etc."""
    present = [n for n in names if n in df.columns]
    if not present:
        df[out] = np.nan
        return df
    result = df[present[0]].copy()
    for name in present[1:]:
        result = result.where(result.notna(), df[name])
    df[out] = pd.to_numeric(result, errors="coerce")
    return df


def _rds_path(stem: str) -> Path:
    return RAW_DIR / f"{stem}.rds"


def download_table(stem: str, *, force: bool = False) -> Path:
    """Fetch one release asset. GitHub 404s are 9-byte bodies — treat as missing."""
    import urllib.request

    dest = _rds_path(stem)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 64 and not force:
        return dest
    url = f"{RELEASE_BASE}/{stem}.rds"
    print(f"downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to download {url}") from exc
    if dest.stat().st_size < 64:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{url} returned an empty/404 body")
    return dest


def read_rds(path: Path) -> pd.DataFrame:
    tables = pyreadr.read_r(str(path))
    frame = tables[None] if None in tables else next(iter(tables.values()))
    return frame


def load_player_table(name: str, *, force_download: bool = False) -> pd.DataFrame:
    path = download_table(f"big5_player_{name}", force=force_download)
    df = read_rds(path)
    if "Season_End_Year" in df.columns:
        df = df.loc[df["Season_End_Year"].between(SEASON_MIN, SEASON_MAX)].copy()
    return df


def load_team_possession(*, force_download: bool = False) -> pd.DataFrame:
    path = download_table("big5_team_possession", force=force_download)
    df = read_rds(path)
    df = df.loc[df["Season_End_Year"].between(SEASON_MIN, SEASON_MAX)].copy()
    team = df.loc[df["Team_or_Opponent"] == "team", JOIN_KEYS + ["Poss"]].copy()
    team = team.rename(columns={"Poss": "team_poss"})
    team["team_poss"] = pd.to_numeric(team["team_poss"], errors="coerce")
    return team.drop_duplicates(JOIN_KEYS)


def _drop_overlapping(frame: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    overlap = [c for c in extra.columns if c in frame.columns and c not in JOIN_KEYS]
    return extra.drop(columns=overlap)


def _prepare_standard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age"] = out["Age"].map(parse_age)
    out["minutes"] = pd.to_numeric(out["Min_Playing"], errors="coerce")
    out["mp"] = pd.to_numeric(out["MP_Playing"], errors="coerce")
    out["starts"] = pd.to_numeric(out["Starts_Playing"], errors="coerce")
    rename = {
        "Gls": "gls",
        "Ast": "ast",
        "PK": "pk",
        "PKatt": "pkatt",
        "CrdY": "crdy",
        "CrdR": "crdr",
        "xG_Expected": "xg",
        "npxG_Expected": "npxg",
        "xAG_Expected": "xag",
        "PrgP_Progression": "prg_p",
        "PrgC_Progression": "prg_c",
        "PrgR_Progression": "prg_r",
        "Player": "player",
        "Nation": "nation",
        "Pos": "pos",
        "Born": "born",
        "Comp": "comp",
        "Url": "url",
        "Squad": "squad",
        "Season_End_Year": "season_end_year",
    }
    out = out.rename(columns=rename)
    keep = [
        "url",
        "season_end_year",
        "squad",
        "player",
        "nation",
        "pos",
        "age",
        "born",
        "comp",
        "mp",
        "starts",
        "minutes",
        "gls",
        "ast",
        "pk",
        "pkatt",
        "crdy",
        "crdr",
        "xg",
        "npxg",
        "xag",
        "prg_p",
        "prg_c",
        "prg_r",
    ]
    return out[keep]


def _prepare_shooting(df: pd.DataFrame) -> pd.DataFrame:
    out = df[JOIN_KEYS + [
        "Sh_Standard",
        "SoT_Standard",
        "Dist_Standard",
        "FK_Standard",
        "npxG_Expected",
        "xG_Expected",
    ]].copy()
    return out.rename(columns={
        "Sh_Standard": "sh",
        "SoT_Standard": "sot",
        "Dist_Standard": "shot_dist",
        "FK_Standard": "fk_shots",
        "npxG_Expected": "npxg_sht",
        "xG_Expected": "xg_sht",
        "Url": "url",
        "Squad": "squad",
        "Season_End_Year": "season_end_year",
    })


def _prepare_passing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    coalesce_columns(out, "Prog", "PrgP", out="pass_prg_p")
    coalesce_columns(out, "xA", "xA_Expected", out="xa")
    keep_src = JOIN_KEYS + [
        "pass_prg_p",
        "xa",
        "Cmp_Total",
        "Att_Total",
        "TotDist_Total",
        "PrgDist_Total",
        "Cmp_Short",
        "Att_Short",
        "Cmp_Medium",
        "Att_Medium",
        "Cmp_Long",
        "Att_Long",
        "KP",
        "Final_Third",
        "PPA",
        "CrsPA",
    ]
    out = out[keep_src].rename(columns={
        "Cmp_Total": "pass_cmp",
        "Att_Total": "pass_att",
        "TotDist_Total": "pass_totdist",
        "PrgDist_Total": "pass_prgdist",
        "Cmp_Short": "pass_cmp_short",
        "Att_Short": "pass_att_short",
        "Cmp_Medium": "pass_cmp_med",
        "Att_Medium": "pass_att_med",
        "Cmp_Long": "pass_cmp_long",
        "Att_Long": "pass_att_long",
        "KP": "kp",
        "Final_Third": "pass_final_third",
        "PPA": "ppa",
        "CrsPA": "crspa",
        "Url": "url",
        "Squad": "squad",
        "Season_End_Year": "season_end_year",
    })
    return out


def _prepare_passing_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df[JOIN_KEYS + ["TB_Pass", "Sw_Pass", "Crs_Pass", "Live_Pass", "Dead_Pass"]].copy()
    return out.rename(columns={
        "TB_Pass": "tb",
        "Sw_Pass": "sw",
        "Crs_Pass": "crs_pass",
        "Live_Pass": "pass_live",
        "Dead_Pass": "pass_dead",
        "Url": "url",
        "Squad": "squad",
        "Season_End_Year": "season_end_year",
    })


def _prepare_possession(df: pd.DataFrame) -> pd.DataFrame:
    cols = JOIN_KEYS + [
        "Touches_Touches",
        "Def Pen_Touches",
        "Def 3rd_Touches",
        "Mid 3rd_Touches",
        "Att 3rd_Touches",
        "Att Pen_Touches",
        "Att_Take",
        "Succ_Take",
        "Tkld_Take",
        "Carries_Carries",
        "TotDist_Carries",
        "PrgDist_Carries",
        "PrgC_Carries",
        "Final_Third_Carries",
        "CPA_Carries",
        "Mis_Carries",
        "Dis_Carries",
        "Rec_Receiving",
        "PrgR_Receiving",
    ]
    out = df[cols].copy()
    return out.rename(columns={
        "Touches_Touches": "touches",
        "Def Pen_Touches": "touches_def_pen",
        "Def 3rd_Touches": "touches_def3",
        "Mid 3rd_Touches": "touches_mid3",
        "Att 3rd_Touches": "touches_att3",
        "Att Pen_Touches": "touches_att_pen",
        "Att_Take": "take_att",
        "Succ_Take": "take_succ",
        "Tkld_Take": "take_tkld",
        "Carries_Carries": "carries",
        "TotDist_Carries": "carry_totdist",
        "PrgDist_Carries": "carry_prgdist",
        "PrgC_Carries": "prgc_poss",
        "Final_Third_Carries": "carry_final_third",
        "CPA_Carries": "cpa",
        "Mis_Carries": "miscontrols",
        "Dis_Carries": "dispossessed",
        "Rec_Receiving": "receives",
        "PrgR_Receiving": "prgr_poss",
        "Url": "url",
        "Squad": "squad",
        "Season_End_Year": "season_end_year",
    })


def _prepare_defense(df: pd.DataFrame) -> pd.DataFrame:
    cols = JOIN_KEYS + [
        "Tkl_Tackles",
        "TklW_Tackles",
        "Def 3rd_Tackles",
        "Mid 3rd_Tackles",
        "Att 3rd_Tackles",
        "Tkl_Challenges",
        "Att_Challenges",
        "Lost_Challenges",
        "Blocks_Blocks",
        "Sh_Blocks",
        "Pass_Blocks",
        "Int",
        "Clr",
        "Err",
    ]
    out = df[cols].copy()
    return out.rename(columns={
        "Tkl_Tackles": "tkl",
        "TklW_Tackles": "tklw",
        "Def 3rd_Tackles": "tkl_def3",
        "Mid 3rd_Tackles": "tkl_mid3",
        "Att 3rd_Tackles": "tkl_att3",
        "Tkl_Challenges": "tkl_vs_dribble",
        "Att_Challenges": "tkl_vs_dribble_att",
        "Lost_Challenges": "tkl_vs_dribble_lost",
        "Blocks_Blocks": "blocks",
        "Sh_Blocks": "sh_blocks",
        "Pass_Blocks": "pass_blocks",
        "Int": "interceptions",
        "Clr": "clearances",
        "Err": "errors",
        "Url": "url",
        "Squad": "squad",
        "Season_End_Year": "season_end_year",
    })


def _prepare_misc(df: pd.DataFrame) -> pd.DataFrame:
    cols = JOIN_KEYS + [
        "Fls",
        "Fld",
        "Off",
        "Crs",
        "Recov",
        "Won_Aerial",
        "Lost_Aerial",
        "PKwon",
        "PKcon",
    ]
    out = df[cols].copy()
    return out.rename(columns={
        "Fls": "fls",
        "Fld": "fld",
        "Off": "offsides",
        "Crs": "crosses",
        "Recov": "recoveries",
        "Won_Aerial": "aerial_won",
        "Lost_Aerial": "aerial_lost",
        "PKwon": "pk_won",
        "PKcon": "pk_con",
        "Url": "url",
        "Squad": "squad",
        "Season_End_Year": "season_end_year",
    })


def _prepare_keepers(df: pd.DataFrame) -> pd.DataFrame:
    cols = JOIN_KEYS + [
        "GA",
        "SoTA",
        "Saves",
        "CS",
        "PKatt_Penalty",
        "PKA_Penalty",
        "PKsv_Penalty",
        "PKm_Penalty",
    ]
    out = df[cols].copy()
    return out.rename(columns={
        "GA": "gk_ga",
        "SoTA": "gk_sota",
        "Saves": "gk_saves",
        "CS": "gk_cs",
        "PKatt_Penalty": "gk_pkatt",
        "PKA_Penalty": "gk_pka",
        "PKsv_Penalty": "gk_pksv",
        "PKm_Penalty": "gk_pkm",
        "Url": "url",
        "Squad": "squad",
        "Season_End_Year": "season_end_year",
    })


def _prepare_keepers_adv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    coalesce_columns(out, "Att_Passes", "Att (GK)_Passes", out="gk_pass_att")
    cols = JOIN_KEYS + [
        "gk_pass_att",
        "PSxG_Expected",
        "PSxG+_per__minus__Expected",
        "Cmp_Launched",
        "Att_Launched",
        "Thr_Passes",
        "Launch_percent_Passes",
        "AvgLen_Passes",
        "Att_Goal",
        "Launch_percent_Goal",
        "AvgLen_Goal",
        "Opp_Crosses",
        "Stp_Crosses",
        "#OPA_Sweeper",
        "AvgDist_Sweeper",
    ]
    out = out[cols].rename(columns={
        "PSxG_Expected": "gk_psxg",
        "PSxG+_per__minus__Expected": "gk_psxg_plus_minus",
        "Cmp_Launched": "gk_launch_cmp",
        "Att_Launched": "gk_launch_att",
        "Thr_Passes": "gk_throws",
        "Launch_percent_Passes": "gk_launch_pct",
        "AvgLen_Passes": "gk_avg_pass_len",
        "Att_Goal": "gk_goal_kick_att",
        "Launch_percent_Goal": "gk_goal_kick_launch_pct",
        "AvgLen_Goal": "gk_avg_goal_kick_len",
        "Opp_Crosses": "gk_crosses_faced",
        "Stp_Crosses": "gk_crosses_stopped",
        "#OPA_Sweeper": "gk_opa",
        "AvgDist_Sweeper": "gk_sweeper_avg_dist",
        "Url": "url",
        "Squad": "squad",
        "Season_End_Year": "season_end_year",
    })
    return out


COUNT_COLS = [
    "mp",
    "starts",
    "minutes",
    "gls",
    "ast",
    "pk",
    "pkatt",
    "crdy",
    "crdr",
    "xg",
    "npxg",
    "xag",
    "prg_p",
    "prg_c",
    "prg_r",
    "sh",
    "sot",
    "fk_shots",
    "pass_prg_p",
    "xa",
    "pass_cmp",
    "pass_att",
    "pass_totdist",
    "pass_prgdist",
    "pass_cmp_short",
    "pass_att_short",
    "pass_cmp_med",
    "pass_att_med",
    "pass_cmp_long",
    "pass_att_long",
    "kp",
    "pass_final_third",
    "ppa",
    "crspa",
    "tb",
    "sw",
    "crs_pass",
    "pass_live",
    "pass_dead",
    "touches",
    "touches_def_pen",
    "touches_def3",
    "touches_mid3",
    "touches_att3",
    "touches_att_pen",
    "take_att",
    "take_succ",
    "take_tkld",
    "carries",
    "carry_totdist",
    "carry_prgdist",
    "prgc_poss",
    "carry_final_third",
    "cpa",
    "miscontrols",
    "dispossessed",
    "receives",
    "prgr_poss",
    "tkl",
    "tklw",
    "tkl_def3",
    "tkl_mid3",
    "tkl_att3",
    "tkl_vs_dribble",
    "tkl_vs_dribble_att",
    "tkl_vs_dribble_lost",
    "blocks",
    "sh_blocks",
    "pass_blocks",
    "interceptions",
    "clearances",
    "errors",
    "fls",
    "fld",
    "offsides",
    "crosses",
    "recoveries",
    "aerial_won",
    "aerial_lost",
    "pk_won",
    "pk_con",
    "gk_ga",
    "gk_sota",
    "gk_saves",
    "gk_cs",
    "gk_pkatt",
    "gk_pka",
    "gk_pksv",
    "gk_pkm",
    "gk_pass_att",
    "gk_psxg",
    "gk_psxg_plus_minus",
    "gk_launch_cmp",
    "gk_launch_att",
    "gk_throws",
    "gk_goal_kick_att",
    "gk_crosses_faced",
    "gk_crosses_stopped",
    "gk_opa",
]

WEIGHTED_COLS = [
    "age",
    "shot_dist",
    "team_poss",
    "gk_launch_pct",
    "gk_avg_pass_len",
    "gk_goal_kick_launch_pct",
    "gk_avg_goal_kick_len",
    "gk_sweeper_avg_dist",
]


def _left_join(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    keys = ["url", "season_end_year", "squad"]
    extra = extra.drop(columns=[c for c in extra.columns if c in base.columns and c not in keys])
    return base.merge(extra, on=keys, how="left")


def aggregate_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse mid-season moves to one row per (url, season).

    Counting stats are summed; rates that arrived as averages are minutes-weighted;
    squad/comp/pos come from the spell with the most minutes.
    """
    work = df.copy()
    work["_min"] = pd.to_numeric(work["minutes"], errors="coerce").fillna(0.0)
    keys = ["url", "season_end_year"]
    grouped = work.groupby(keys, sort=False)

    n_spells = grouped.size().rename("n_spells")
    identity_idx = work.groupby(keys)["_min"].idxmax()
    identity = work.loc[identity_idx, [
        "url",
        "season_end_year",
        "player",
        "nation",
        "born",
        "squad",
        "comp",
        "pos",
    ]].set_index(keys)

    present_counts = [c for c in COUNT_COLS if c in work.columns]
    sums = grouped[present_counts].sum(min_count=1)

    weighted_parts = []
    for col in WEIGHTED_COLS:
        if col not in work.columns:
            continue
        tmp = work[["url", "season_end_year", "_min", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.loc[tmp[col].notna() & (tmp["_min"] > 0)]
        if tmp.empty:
            continue
        tmp["_w"] = tmp["_min"] * tmp[col]
        g = tmp.groupby(keys)
        weighted_parts.append((g["_w"].sum() / g["_min"].sum()).rename(col))

    def _join_squads(series: pd.Series) -> str:
        # Preserve minutes-desc order for the representative squads string.
        return " | ".join(dict.fromkeys(series.tolist()))

    squads = (
        work.sort_values("_min", ascending=False)
        .groupby(keys, sort=False)["squad"]
        .agg(_join_squads)
        .rename("squads")
    )

    def _join_pos(series: pd.Series) -> str:
        seen: list[str] = []
        for pos in series.tolist():
            for token in parse_positions(pos):
                if token not in seen:
                    seen.append(token)
        return ",".join(seen)

    pos_combined = (
        work.sort_values("_min", ascending=False)
        .groupby(keys, sort=False)["pos"]
        .agg(_join_pos)
        .rename("pos_combined")
    )

    out = identity.join(n_spells).join(squads).join(pos_combined).join(sums)
    for part in weighted_parts:
        out = out.join(part, how="left")
    out["pos"] = out["pos_combined"].where(out["pos_combined"].fillna("") != "", out["pos"])
    out = out.drop(columns=["pos_combined"])
    out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce")
    return out.reset_index()


def build_player_seasons(*, force_download: bool = False) -> pd.DataFrame:
    print("loading standard (backbone)")
    standard = _prepare_standard(load_player_table("standard", force_download=force_download))

    print("loading shooting / passing / possession / defense / misc / keepers")
    shooting = _prepare_shooting(load_player_table("shooting", force_download=force_download))
    passing = _prepare_passing(load_player_table("passing", force_download=force_download))
    passing_types = _prepare_passing_types(
        load_player_table("passing_types", force_download=force_download)
    )
    possession = _prepare_possession(load_player_table("possession", force_download=force_download))
    defense = _prepare_defense(load_player_table("defense", force_download=force_download))
    misc = _prepare_misc(load_player_table("misc", force_download=force_download))
    keepers = _prepare_keepers(load_player_table("keepers", force_download=force_download))
    keepers_adv = _prepare_keepers_adv(
        load_player_table("keepers_adv", force_download=force_download)
    )
    team_poss = load_team_possession(force_download=force_download).rename(
        columns={"Url": "url", "Squad": "squad", "Season_End_Year": "season_end_year"}
    )
    # team table Url is a *team* page — do not join on it
    team_poss = team_poss.drop(columns=["url"], errors="ignore")

    frame = standard
    for extra in (shooting, passing, passing_types, possession, defense, misc, keepers, keepers_adv):
        frame = _left_join(frame, extra)

    frame = frame.merge(team_poss, on=["squad", "season_end_year"], how="left")

    # Prefer shooting xG if standard is missing a cell (rare).
    if "npxg_sht" in frame.columns:
        frame["npxg"] = frame["npxg"].where(frame["npxg"].notna(), frame["npxg_sht"])
        frame["xg"] = frame["xg"].where(frame["xg"].notna(), frame["xg_sht"])
        frame = frame.drop(columns=["npxg_sht", "xg_sht"], errors="ignore")

    print(f"pre-agg rows: {len(frame):,}  duplicate player-seasons: "
          f"{frame.duplicated(['url', 'season_end_year']).sum():,}")
    frame = aggregate_transfers(frame)

    frame["player_id"] = [
        player_season_id(u, y) for u, y in zip(frame["url"], frame["season_end_year"])
    ]
    frame["fbref_id"] = frame["url"].map(fbref_player_id)
    frame["season"] = frame["season_end_year"].map(lambda y: season_label(int(y)))
    frame["primary_pos"] = frame["pos"].map(primary_position)
    frame["is_gk"] = frame["primary_pos"].eq("GK") | frame["pos"].fillna("").str.contains("GK")
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce")

    # Canonical progressive carries / receptions: standard is complete; fall back to possession.
    if "prgc_poss" in frame.columns:
        frame["prg_c"] = frame["prg_c"].where(frame["prg_c"].notna(), frame["prgc_poss"])
    if "prgr_poss" in frame.columns:
        frame["prg_r"] = frame["prg_r"].where(frame["prg_r"].notna(), frame["prgr_poss"])

    ordered = [
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
        "born",
        "nation",
        "mp",
        "starts",
        "minutes",
        "team_poss",
    ]
    rest = [c for c in frame.columns if c not in ordered]
    frame = frame[ordered + rest].sort_values(
        ["season_end_year", "comp", "player"]
    ).reset_index(drop=True)
    return frame


def run(*, force_download: bool = False, output: Path | None = None) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_player_seasons(force_download=force_download)
    dest = output or PLAYER_SEASONS_PARQUET
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    print(f"wrote {dest}  rows={len(frame):,}  cols={frame.shape[1]}")
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest FBref Big 5 player-seasons")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    run(force_download=args.force_download, output=args.output)


if __name__ == "__main__":
    main()
