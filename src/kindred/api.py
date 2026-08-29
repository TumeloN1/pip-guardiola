"""FastAPI retrieval service. Default metric is z-scored cosine; swapped to the
learned encoder once artifacts/encoder.npz exists.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from kindred.features import FEATURE_GROUPS, GK_GROUPS
from kindred.paths import EMBEDDINGS_NPZ, ENCODER_NPZ, EVAL_JSON, FEATURES_PARQUET, GMM_NPZ, PROJECTION_NPZ
from kindred.similarity import (
    DEFAULT_GK_WEIGHTS,
    DEFAULT_WEIGHTS,
    MatrixIndex,
    build_index,
    group_match_explanations,
    similar,
    weighted_matrix,
)

API_PORT = 8317

app = FastAPI(title="Kindred", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_state() -> dict[str, Any]:
    features = pd.read_parquet(FEATURES_PARQUET)
    outfield = build_index(features, "outfield")
    keeper = build_index(features, "keeper")
    encoder = None
    encoder_gk = None
    embeddings: dict[str, np.ndarray] = {}
    if ENCODER_NPZ.exists():
        from kindred.model import load_encoder

        encoder = load_encoder(ENCODER_NPZ)
    gk_path = ENCODER_NPZ.with_name("encoder_gk.npz")
    if gk_path.exists():
        from kindred.model import load_encoder

        encoder_gk = load_encoder(gk_path)
    if EMBEDDINGS_NPZ.exists():
        blob = np.load(EMBEDDINGS_NPZ, allow_pickle=False)
        for key in blob.files:
            embeddings[key] = blob[key]
    gmm = None
    if GMM_NPZ.exists():
        from kindred.cluster import load_gmm

        gmm = load_gmm(GMM_NPZ)
    projection = None
    if PROJECTION_NPZ.exists():
        blob = np.load(PROJECTION_NPZ, allow_pickle=False)
        projection = {k: blob[k] for k in blob.files}
    eval_report = json.loads(EVAL_JSON.read_text()) if EVAL_JSON.exists() else {}
    return {
        "features": features,
        "indexes": {"outfield": outfield, "keeper": keeper},
        "encoder": encoder,
        "encoder_gk": encoder_gk,
        "embeddings": embeddings,
        "gmm": gmm,
        "projection": projection,
        "eval": eval_report,
    }


@lru_cache(maxsize=1)
def state() -> dict[str, Any]:
    return _load_state()


def reload_state() -> None:
    state.cache_clear()
    state()


def _index_for(player_id: str) -> tuple[MatrixIndex, str]:
    st = state()
    for role, index in st["indexes"].items():
        if player_id in index.id_to_row():
            return index, role
    raise HTTPException(404, f"unknown player-season {player_id}")


def _row_payload(index: MatrixIndex, i: int, extra: dict | None = None) -> dict:
    body = {
        "id": str(index.ids[i]),
        "player": str(index.names[i]),
        "season": str(index.seasons[i]),
        "season_end_year": int(index.season_end_year[i]),
        "squad": str(index.squads[i]),
        "comp": str(index.comps[i]),
        "pos": str(index.positions[i]),
        "primary_pos": str(index.primary_pos[i]),
        "minutes": float(index.minutes[i]),
        "fbref_id": str(index.fbref_ids[i]),
        "role": index.role,
    }
    if extra:
        body.update(extra)
    return body


def _weights_are_default(index: MatrixIndex, weights: dict[str, float] | None) -> bool:
    defaults = DEFAULT_GK_WEIGHTS if index.role == "keeper" else DEFAULT_WEIGHTS
    if not weights:
        return True
    return all(abs(float(weights.get(k, 1.0)) - 1.0) < 1e-6 for k in defaults)


def _cached_embeddings(index: MatrixIndex) -> np.ndarray | None:
    st = state()
    key = "outfield" if index.role == "outfield" else "keeper"
    if key not in st["embeddings"] or f"{key}_ids" not in st["embeddings"]:
        return None
    ids = np.asarray(st["embeddings"][f"{key}_ids"]).astype(str)
    emb = st["embeddings"][key]
    lookup = {pid: i for i, pid in enumerate(ids)}
    try:
        order = np.array([lookup[str(pid)] for pid in index.ids])
    except KeyError:
        return None
    return emb[order]


def _active_matrix(index: MatrixIndex, weights: dict[str, float] | None) -> np.ndarray:
    st = state()
    scaled = weighted_matrix(index, weights)
    if index.role == "keeper":
        from kindred.similarity import pca_whiten

        return pca_whiten(scaled, n_components=min(12, scaled.shape[1]))
    encoder = st["encoder"]
    if encoder is None:
        return scaled
    if _weights_are_default(index, weights):
        cached = _cached_embeddings(index)
        if cached is not None:
            return cached
    from kindred.model import encode_matrix

    return encode_matrix(encoder, scaled)


@app.get("/api/health")
def health() -> dict:
    st = state()
    return {
        "ok": True,
        "n_outfield": len(st["indexes"]["outfield"].ids),
        "n_keeper": len(st["indexes"]["keeper"].ids),
        "metric": "learned" if st["encoder"] is not None else "zscore_cosine",
        "has_gmm": st["gmm"] is not None,
    }


@app.get("/api/meta")
def meta() -> dict:
    st = state()
    features: pd.DataFrame = st["features"]
    comps = sorted(features["comp"].dropna().unique().tolist())
    seasons = sorted(int(y) for y in features["season_end_year"].unique())
    return {
        "comps": comps,
        "season_end_years": seasons,
        "positions": ["GK", "DF", "MF", "FW"],
        "groups": FEATURE_GROUPS,
        "gk_groups": GK_GROUPS,
        "default_weights": DEFAULT_WEIGHTS,
        "default_gk_weights": DEFAULT_GK_WEIGHTS,
        "metric": "learned" if st["encoder"] is not None else "zscore_cosine",
        "eval": st["eval"],
        "examples": [
            {"id": "e46012d4-2020", "label": "Kevin De Bruyne · 2019-20"},
            {"id": "e342ad68-2020", "label": "Mohamed Salah · 2019-20"},
            {"id": "e06683ca-2019", "label": "Virgil van Dijk · 2018-19"},
            {"id": "1f44ac21-2023", "label": "Erling Haaland · 2022-23"},
        ],
    }


@app.get("/api/players")
def search_players(q: str = Query("", min_length=0), limit: int = 20) -> list[dict]:
    st = state()
    features: pd.DataFrame = st["features"]
    needle = q.strip().lower()
    frame = features
    if needle:
        frame = features.loc[features["player"].str.lower().str.contains(needle, na=False, regex=False)]
    frame = frame.sort_values(["minutes", "season_end_year"], ascending=False).head(limit)
    rows = []
    for _, r in frame.iterrows():
        rows.append({
            "id": r["player_id"],
            "player": r["player"],
            "season": r["season"],
            "season_end_year": int(r["season_end_year"]),
            "squad": r["squad"],
            "comp": r["comp"],
            "pos": r["pos"],
            "minutes": float(r["minutes"]),
            "role": r["role"],
        })
    return rows


@app.get("/api/players/{player_id}")
def get_player(player_id: str) -> dict:
    index, _role = _index_for(player_id)
    i = index.id_to_row()[player_id]
    return _row_payload(index, i)


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


@app.get("/api/players/{player_id}/similar")
def get_similar(
    player_id: str,
    era_start: int | None = 2018,
    era_end: int | None = 2025,
    comps: str | None = None,
    positions: str | None = None,
    min_minutes: float = 900,
    k: int = 15,
    weights: str | None = None,
) -> dict:
    index, _role = _index_for(player_id)
    weight_map = json.loads(weights) if weights else {}
    matrix = _active_matrix(index, weight_map)
    hits = similar(
        index,
        player_id,
        k=k,
        weights=weight_map,
        era_start=era_start,
        era_end=era_end,
        comps=_parse_csv(comps),
        positions=_parse_csv(positions),
        min_minutes=min_minutes,
        matrix=matrix,
    )
    explained = []
    st = state()
    for hit in hits:
        if index.role == "keeper" or st["encoder"] is None:
            hit["groups"] = group_match_explanations(index, player_id, hit["player_id"], weight_map)
        else:
            from kindred.cluster import gradient_group_explanations

            hit["groups"] = gradient_group_explanations(
                st["encoder"], index, player_id, hit["player_id"], weight_map
            )
        explained.append(hit)
    qi = index.id_to_row()[player_id]
    return {"query": _row_payload(index, qi), "results": explained, "metric": state()["encoder"] and "learned" or "zscore_cosine"}


@app.get("/api/players/{player_id}/profile")
def get_profile(player_id: str) -> dict:
    index, role = _index_for(player_id)
    i = index.id_to_row()[player_id]
    raw = index.raw
    if raw is None:
        raise HTTPException(500, "raw features missing")
    # Percentile vs everyone in this role.
    percentiles = {}
    for j, name in enumerate(index.feature_names):
        col = raw[:, j]
        finite = col[np.isfinite(col)]
        value = float(raw[i, j])
        if finite.size == 0 or not np.isfinite(value):
            percentiles[name] = None
        else:
            percentiles[name] = float((finite <= value).mean() * 100.0)
    archetypes = []
    st = state()
    if st["gmm"] is not None and role == "outfield" and "outfield" in st["embeddings"]:
        from kindred.cluster import responsibilities_for

        ids = np.asarray(st["embeddings"]["outfield_ids"]).astype(str)
        loc = np.where(ids == player_id)[0]
        if loc.size:
            archetypes = responsibilities_for(st["gmm"], st["embeddings"]["outfield"][int(loc[0])])
    radar_keys = {
        "outfield": [
            "npxg_p90",
            "xag_p90",
            "kp_p90",
            "prg_p_p90",
            "prg_c_p90",
            "take_att_p90",
            "padj_tkl",
            "padj_int",
            "aerial_p90",
            "touch_att3_share",
        ],
        "keeper": [
            "gk_save_share",
            "gk_psxg_plus_minus_p90",
            "gk_launch_pct",
            "gk_avg_pass_len",
            "gk_opa_p90",
            "gk_cross_stop_share",
        ],
    }[role]
    radar = [{"feature": f, "percentile": percentiles.get(f), "label": _pretty(f)} for f in radar_keys]
    return {
        "player": _row_payload(index, i),
        "percentiles": percentiles,
        "radar": radar,
        "archetypes": archetypes,
        "groups": list(index.groups.keys()),
    }


def _pretty(name: str) -> str:
    mapping = {
        "npxg_p90": "npxG/90",
        "xag_p90": "xAG/90",
        "kp_p90": "Key passes/90",
        "prg_p_p90": "Prog. passes/90",
        "prg_c_p90": "Prog. carries/90",
        "take_att_p90": "Take-ons/90",
        "padj_tkl": "PAdj tackles",
        "padj_int": "PAdj interceptions",
        "aerial_p90": "Aerials won/90",
        "touch_att3_share": "Att. third touches",
        "gk_save_share": "Save %",
        "gk_psxg_plus_minus_p90": "PSxG +/- /90",
        "gk_launch_pct": "Launch %",
        "gk_avg_pass_len": "Avg pass length",
        "gk_opa_p90": "Sweeper actions/90",
        "gk_cross_stop_share": "Crosses stopped",
    }
    return mapping.get(name, name)


@app.get("/api/projection")
def get_projection(role: str = "outfield", sample: int = 2500) -> dict:
    st = state()
    index: MatrixIndex = st["indexes"][role]
    if st["projection"] is not None and f"{role}_xy" in st["projection"]:
        xy = st["projection"][f"{role}_xy"]
        ids = st["projection"][f"{role}_ids"]
        lookup = {str(i): k for k, i in enumerate(ids.astype(str))}
        points = []
        for n, pid in enumerate(index.ids.astype(str)):
            j = lookup.get(pid)
            if j is None:
                continue
            points.append(_row_payload(index, n, {"x": float(xy[j, 0]), "y": float(xy[j, 1])}))
        source = "learned"
    else:
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2, random_state=0).fit_transform(index.features)
        points = [
            _row_payload(index, n, {"x": float(xy[n, 0]), "y": float(xy[n, 1])})
            for n in range(len(index.ids))
        ]
        source = "pca"
    if sample and len(points) > sample:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(points), size=sample, replace=False)
        points = [points[i] for i in sorted(pick.tolist())]
    return {"points": points, "source": source}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=API_PORT)
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run("kindred.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
