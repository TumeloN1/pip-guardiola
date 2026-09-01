"""Nearest-neighbour retrieval over z-scored (or embedding) vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kindred.features import FEATURE_GROUPS, GK_FEATURES, GK_GROUPS, OUTFIELD_FEATURES


DEFAULT_WEIGHTS = {name: 1.0 for name in FEATURE_GROUPS}
DEFAULT_GK_WEIGHTS = {name: 1.0 for name in GK_GROUPS}


@dataclass
class MatrixIndex:
    ids: np.ndarray
    names: np.ndarray
    seasons: np.ndarray
    season_end_year: np.ndarray
    comps: np.ndarray
    squads: np.ndarray
    positions: np.ndarray
    primary_pos: np.ndarray
    minutes: np.ndarray
    fbref_ids: np.ndarray
    features: np.ndarray  # (n, d) already z-scored, possibly unweighted
    feature_names: list[str]
    groups: dict[str, list[str]]
    role: str
    raw: np.ndarray | None = None

    def id_to_row(self) -> dict[str, int]:
        return {pid: i for i, pid in enumerate(self.ids.tolist())}


def _z_cols(names: list[str]) -> list[str]:
    return [f"z_{n}" for n in names]


def _raw_cols(names: list[str]) -> list[str]:
    return [f"raw_{n}" for n in names]


def build_index(features: pd.DataFrame, role: str = "outfield") -> MatrixIndex:
    subset = features.loc[features["role"] == role].reset_index(drop=True)
    names = OUTFIELD_FEATURES if role == "outfield" else GK_FEATURES
    groups = FEATURE_GROUPS if role == "outfield" else GK_GROUPS
    z = subset[_z_cols(names)].to_numpy(dtype=np.float32)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    raw_cols = _raw_cols(names)
    raw = subset[raw_cols].to_numpy(dtype=np.float32) if all(c in subset.columns for c in raw_cols) else None
    return MatrixIndex(
        ids=subset["player_id"].to_numpy(),
        names=subset["player"].to_numpy(),
        seasons=subset["season"].to_numpy(),
        season_end_year=subset["season_end_year"].to_numpy(),
        comps=subset["comp"].to_numpy(),
        squads=subset["squad"].to_numpy(),
        positions=subset["pos"].to_numpy(),
        primary_pos=subset["primary_pos"].to_numpy(),
        minutes=subset["minutes"].to_numpy(),
        fbref_ids=subset["fbref_id"].to_numpy(),
        features=z,
        feature_names=names,
        groups=groups,
        role=role,
        raw=raw,
    )


def group_scale_vector(index: MatrixIndex, weights: dict[str, float] | None) -> np.ndarray:
    scale = np.ones(len(index.feature_names), dtype=np.float32)
    if not weights:
        return scale
    name_to_i = {n: i for i, n in enumerate(index.feature_names)}
    defaults = DEFAULT_WEIGHTS if index.role == "outfield" else DEFAULT_GK_WEIGHTS
    for group, cols in index.groups.items():
        w = float(weights.get(group, defaults.get(group, 1.0)))
        for col in cols:
            scale[name_to_i[col]] = w
    return scale


def weighted_matrix(index: MatrixIndex, weights: dict[str, float] | None) -> np.ndarray:
    return index.features * group_scale_vector(index, weights)[None, :]


def l2_normalize(mat: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=axis, keepdims=True)
    return mat / np.maximum(norms, eps)


def pca_whiten(mat: np.ndarray, n_components: int | None = None, eps: float = 1e-5) -> np.ndarray:
    """PCA-whitened representation used as a cosine baseline."""
    centered = mat - mat.mean(axis=0, keepdims=True)
    n_components = n_components or min(mat.shape)
    # SVD on the centered matrix; whitening divides by singular values.
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(n_components, vt.shape[0])
    return (u[:, :k] * np.sqrt(mat.shape[0] - 1))[:, :k]  # already whitened via U * sqrt(n-1)
    # Equivalent to centered @ vt[:k].T / (s[:k] + eps) * sqrt(n-1)


def cosine_scores(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    q = l2_normalize(query.reshape(1, -1))
    g = l2_normalize(gallery)
    return (g @ q.T).ravel()


def apply_filters(
    index: MatrixIndex,
    *,
    era_start: int | None = None,
    era_end: int | None = None,
    comps: list[str] | None = None,
    positions: list[str] | None = None,
    min_minutes: float | None = None,
    exclude_self_fbref: str | None = None,
    exclude_row: int | None = None,
) -> np.ndarray:
    mask = np.ones(len(index.ids), dtype=bool)
    if era_start is not None:
        mask &= index.season_end_year >= era_start
    if era_end is not None:
        mask &= index.season_end_year <= era_end
    if comps:
        mask &= np.isin(index.comps, np.array(comps))
    if positions:
        pos_mask = np.zeros(len(index.ids), dtype=bool)
        for token in positions:
            pos_mask |= np.array([token in parse_pos(p) for p in index.positions])
        mask &= pos_mask
    if min_minutes is not None:
        mask &= index.minutes >= min_minutes
    if exclude_self_fbref:
        mask &= index.fbref_ids != exclude_self_fbref
    if exclude_row is not None:
        mask[exclude_row] = False
    return mask


def parse_pos(pos: object) -> list[str]:
    if pos is None:
        return []
    return [p.strip() for p in str(pos).split(",") if p.strip()]


def similar(
    index: MatrixIndex,
    player_id: str,
    *,
    k: int = 15,
    weights: dict[str, float] | None = None,
    era_start: int | None = None,
    era_end: int | None = None,
    comps: list[str] | None = None,
    positions: list[str] | None = None,
    min_minutes: float | None = None,
    exclude_self: bool = True,
    matrix: np.ndarray | None = None,
) -> list[dict]:
    lookup = index.id_to_row()
    if player_id not in lookup:
        raise KeyError(player_id)
    q = lookup[player_id]
    mat = matrix if matrix is not None else weighted_matrix(index, weights)
    mask = apply_filters(
        index,
        era_start=era_start,
        era_end=era_end,
        comps=comps,
        positions=positions,
        min_minutes=min_minutes,
        exclude_self_fbref=index.fbref_ids[q] if exclude_self else None,
        exclude_row=q,
    )
    if not mask.any():
        return []
    scores = cosine_scores(mat[q], mat[mask])
    order = np.argsort(-scores)[:k]
    rows = np.flatnonzero(mask)[order]
    results = []
    for rank, (row, score) in enumerate(zip(rows, scores[order]), start=1):
        results.append({
            "rank": rank,
            "player_id": str(index.ids[row]),
            "player": str(index.names[row]),
            "season": str(index.seasons[row]),
            "season_end_year": int(index.season_end_year[row]),
            "squad": str(index.squads[row]),
            "comp": str(index.comps[row]),
            "pos": str(index.positions[row]),
            "minutes": float(index.minutes[row]),
            "fbref_id": str(index.fbref_ids[row]),
            "similarity": float(score),
        })
    return results


def group_match_explanations(
    index: MatrixIndex,
    query_id: str,
    candidate_id: str,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Cosine contribution of each feature group (baseline metric)."""
    lookup = index.id_to_row()
    qi, ci = lookup[query_id], lookup[candidate_id]
    scale = group_scale_vector(index, weights)
    q = index.features[qi] * scale
    c = index.features[ci] * scale
    qn = q / max(np.linalg.norm(q), 1e-8)
    cn = c / max(np.linalg.norm(c), 1e-8)
    contrib = qn * cn
    name_to_i = {n: i for i, n in enumerate(index.feature_names)}
    rows = []
    for group, cols in index.groups.items():
        idx = [name_to_i[col] for col in cols]
        rows.append({
            "group": group,
            "score": float(contrib[idx].sum()),
            "weight": float(scale[idx[0]]) if idx else 1.0,
        })
    rows.sort(key=lambda r: -r["score"])
    return rows
