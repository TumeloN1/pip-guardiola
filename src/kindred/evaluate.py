"""Baselines + adjacent-season retrieval eval.

Headline metric: for each player-season, rank every other player-season and
record where the same player's neighbouring season lands. Report MRR and
recall@10. Consecutive-season positives are held out of any later contrastive
training, so this number is not contaminated.

Two baselines must be beaten by the JAX encoder:
  1. z-scored cosine
  2. PCA-whitened cosine
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kindred.paths import EVAL_JSON, FEATURES_PARQUET, ROOT
from kindred.similarity import (
    MatrixIndex,
    build_index,
    cosine_scores,
    l2_normalize,
    pca_whiten,
)

FACE_VALIDITY_PATH = ROOT / "tests" / "face_validity.json"


def _adjacent_pairs(index: MatrixIndex) -> list[tuple[int, int]]:
    """Pairs of rows (query, neighbour) for the same player in consecutive seasons."""
    by_player: dict[str, list[int]] = {}
    for i, fbref in enumerate(index.fbref_ids.tolist()):
        by_player.setdefault(fbref, []).append(i)
    pairs: list[tuple[int, int]] = []
    for rows in by_player.values():
        rows = sorted(rows, key=lambda i: int(index.season_end_year[i]))
        years = [int(index.season_end_year[i]) for i in rows]
        for a, b in zip(range(len(rows) - 1), range(1, len(rows))):
            if years[b] - years[a] == 1:
                pairs.append((rows[a], rows[b]))
                pairs.append((rows[b], rows[a]))
    return pairs


def _mrr_recall(ranks: list[int], k: int = 10) -> dict[str, float]:
    if not ranks:
        return {"n": 0, "mrr": 0.0, "recall_at_10": 0.0, "median_rank": None}
    arr = np.asarray(ranks, dtype=float)
    return {
        "n": int(len(arr)),
        "mrr": float(np.mean(1.0 / arr)),
        "recall_at_10": float(np.mean(arr <= k)),
        "median_rank": float(np.median(arr)),
        "mean_rank": float(np.mean(arr)),
    }


def adjacent_season_retrieval(index: MatrixIndex, matrix: np.ndarray, k: int = 10) -> dict:
    pairs = _adjacent_pairs(index)
    gallery = l2_normalize(matrix)
    ranks: list[int] = []
    for qi, ni in pairs:
        # Rank among everyone except the query row. Neighbour may sit anywhere.
        mask = np.ones(len(index.ids), dtype=bool)
        mask[qi] = False
        scores = gallery[mask] @ gallery[qi]
        order = np.argsort(-scores)
        # Map back to original row indices
        ranked_rows = np.flatnonzero(mask)[order]
        loc = np.where(ranked_rows == ni)[0]
        if loc.size:
            ranks.append(int(loc[0]) + 1)
    return _mrr_recall(ranks, k=k)


def position_purity_at_k(index: MatrixIndex, matrix: np.ndarray, k: int = 10) -> dict:
    gallery = l2_normalize(matrix)
    sims = gallery @ gallery.T
    np.fill_diagonal(sims, -np.inf)
    hits = []
    for i in range(len(index.ids)):
        top = np.argpartition(-sims[i], kth=min(k, sims.shape[1] - 1))[:k]
        # If more than k, sort those
        top = top[np.argsort(-sims[i, top])][:k]
        qpos = str(index.primary_pos[i])
        n_match = sum(str(index.primary_pos[j]) == qpos for j in top)
        hits.append(n_match / max(len(top), 1))
    return {
        "k": k,
        "mean_purity": float(np.mean(hits)),
        "n": int(len(hits)),
    }


def _resolve_player(features: pd.DataFrame, spec: dict) -> str | None:
    hit = features.loc[
        features["player"].eq(spec["player"])
        & features["season_end_year"].eq(spec["season_end_year"])
    ]
    if spec.get("squad"):
        hit = hit.loc[hit["squad"].eq(spec["squad"])]
    if hit.empty:
        return None
    return str(hit.iloc[0]["player_id"])


def face_validity(
    index: MatrixIndex,
    matrix: np.ndarray,
    features: pd.DataFrame,
    cases: list[dict],
) -> dict:
    lookup = index.id_to_row()
    gallery = l2_normalize(matrix)
    results = []
    n_pass = 0
    for case in cases:
        qid = _resolve_player(features, case["query"])
        if qid is None or qid not in lookup:
            results.append({"query": case["query"], "status": "missing_query"})
            continue
        qi = lookup[qid]
        expected_ids = []
        for spec in case["expected_any"]:
            eid = _resolve_player(features, spec)
            if eid is not None and eid in lookup:
                expected_ids.append(eid)
        if not expected_ids:
            results.append({"query": case["query"], "status": "missing_expected"})
            continue
        mask = np.ones(len(index.ids), dtype=bool)
        mask[qi] = False
        # Face-validity is about analogue retrieval, so drop the same player.
        mask &= index.fbref_ids != index.fbref_ids[qi]
        scores = gallery[mask] @ gallery[qi]
        order = np.argsort(-scores)
        ranked = np.array(index.ids[mask][order], dtype=str)
        k = int(case.get("k", 25))
        top = set(ranked[:k].tolist())
        found = [eid for eid in expected_ids if eid in top]
        ranks = []
        for eid in expected_ids:
            loc = np.where(ranked == eid)[0]
            ranks.append(int(loc[0]) + 1 if loc.size else None)
        ok = bool(found)
        n_pass += int(ok)
        results.append({
            "query": case["query"],
            "status": "pass" if ok else "fail",
            "k": k,
            "found": found,
            "ranks": ranks,
        })
    return {
        "n": len(cases),
        "n_pass": n_pass,
        "pass_rate": n_pass / max(len(cases), 1),
        "cases": results,
    }


def evaluate_baselines(features: pd.DataFrame) -> dict:
    report: dict = {"roles": {}}
    for role in ("outfield", "keeper"):
        index = build_index(features, role=role)
        z = index.features
        pca = pca_whiten(z, n_components=min(32, z.shape[1]))
        role_report = {
            "n": int(len(index.ids)),
            "dims": int(z.shape[1]),
            "zscore_cosine": {
                "adjacent": adjacent_season_retrieval(index, z),
                "position_purity_at_10": position_purity_at_k(index, z, k=10),
            },
            "pca_whitened_cosine": {
                "adjacent": adjacent_season_retrieval(index, pca),
                "position_purity_at_10": position_purity_at_k(index, pca, k=10),
            },
        }
        report["roles"][role] = role_report
    return report


def _maybe_learned(features: pd.DataFrame, report: dict) -> None:
    from kindred.paths import ENCODER_NPZ
    from kindred.model import encode_matrix, load_encoder

    mapping = {
        "outfield": ENCODER_NPZ,
        "keeper": ENCODER_NPZ.with_name("encoder_gk.npz"),
    }
    for role, path in mapping.items():
        if not path.exists():
            continue
        bundle = load_encoder(path)
        index = build_index(features, role=role)
        emb = encode_matrix(bundle, index.features)
        report["roles"][role]["learned"] = {
            "adjacent": adjacent_season_retrieval(index, emb),
            "position_purity_at_10": position_purity_at_k(index, emb, k=10),
        }
        if role == "outfield":
            cases = json.loads(FACE_VALIDITY_PATH.read_text()) if FACE_VALIDITY_PATH.exists() else []
            report["face_validity_learned"] = face_validity(index, emb, features, cases)


def run(*, output: Path | None = None) -> dict:
    features = pd.read_parquet(FEATURES_PARQUET)
    report = evaluate_baselines(features)
    cases = json.loads(FACE_VALIDITY_PATH.read_text()) if FACE_VALIDITY_PATH.exists() else []
    outfield = build_index(features, role="outfield")
    report["face_validity_zscore"] = face_validity(
        outfield, outfield.features, features, cases
    )
    _maybe_learned(features, report)
    dest = output or EVAL_JSON
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    for role, body in report["roles"].items():
        print(f"\n== {role} n={body['n']} d={body['dims']} ==")
        for name in ("zscore_cosine", "pca_whitened_cosine", "learned"):
            if name not in body:
                continue
            adj = body[name]["adjacent"]
            pur = body[name]["position_purity_at_10"]
            print(
                f"  {name:22s}  MRR={adj['mrr']:.3f}  R@10={adj['recall_at_10']:.3f}  "
                f"med_rank={adj['median_rank']:.1f}  pos_purity@10={pur['mean_purity']:.3f}"
            )
    fv = report.get("face_validity_zscore", {})
    print(f"\nface validity (z-score): {fv.get('n_pass')}/{fv.get('n')} passed")
    fvl = report.get("face_validity_learned")
    if fvl:
        print(f"face validity (learned): {fvl.get('n_pass')}/{fvl.get('n')} passed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Kindred baselines")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    run(output=args.output)


if __name__ == "__main__":
    main()
