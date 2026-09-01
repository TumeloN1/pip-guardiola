"""Named football-style archetypes over outfield z-features.

Retrieval stays contrastive / cosine. These labels are a fan-facing overlay.

We fit a 16-component diagonal GMM on the already z-scored outfield matrix,
then assign each component to a curated prototype (Hungarian matching) so the
names stay stable and readable — never raw FBref codes like cpa_p90.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

from kindred.features import FEATURE_GROUPS, GK_FEATURES, OUTFIELD_FEATURES
from kindred.paths import EMBEDDINGS_NPZ, FEATURES_PARQUET, GMM_NPZ, PROJECTION_NPZ

N_ARCHETYPES = 16

# Prototypes: high / low z-score features a season of that style should show.
ARCHETYPE_CATALOG: list[dict] = [
    {
        "name": "Deep-lying playmaker",
        "blurb": "Dictates from deep: progressive passes, switches, and key balls without hunting goals.",
        "high": ["prg_p_p90", "pass_long_share", "kp_p90", "pass_att_p90", "prg_pass_share"],
        "low": ["npxg_p90", "touch_att_pen_share", "gls_p90"],
    },
    {
        "name": "Box-to-box midfielder",
        "blurb": "Covers both boxes: progressive carries, tackles, and a share of chance creation.",
        "high": ["prg_c_p90", "padj_tkl", "xag_p90", "carries_p90", "touch_mid3_share"],
        "low": ["touch_def_pen_share"],
    },
    {
        "name": "Destroyer",
        "blurb": "Wins the ball and stays compact: tackles, interceptions, blocks, clearances.",
        "high": ["padj_tkl", "padj_int", "padj_blocks", "padj_clr", "tkl_def_share"],
        "low": ["npxg_p90", "prg_c_p90", "touch_att_pen_share"],
    },
    {
        "name": "Ball-playing centre-back",
        "blurb": "Starts attacks from the back: progressive passing and switches, not just clearances.",
        "high": ["prg_p_p90", "pass_long_share", "pass_att_p90", "carries_p90", "touch_def3_share"],
        "low": ["npxg_p90", "touch_att_pen_share"],
    },
    {
        "name": "Stopper",
        "blurb": "Old-school centre-back: aerials, clearances, and blocks in the defensive box.",
        "high": ["padj_clr", "aerial_p90", "padj_blocks", "touch_def_pen_share", "aerial_win_share"],
        "low": ["prg_p_p90", "prg_c_p90"],
    },
    {
        "name": "Overlapping full-back",
        "blurb": "Attacks the flank: progressive carries, crosses, and touches in the attacking third.",
        "high": ["prg_c_p90", "touch_att3_share", "carries_p90", "prg_r_p90", "tkl_att_share"],
        "low": ["npxg_p90"],
    },
    {
        "name": "Inverted full-back",
        "blurb": "Tucks inside to pass: progressive passing volume over crossing and wide carries.",
        "high": ["prg_p_p90", "pass_att_p90", "touch_mid3_share", "carries_p90", "pass_short_share"],
        "low": ["take_att_p90", "touch_att_pen_share"],
    },
    {
        "name": "Wing-back",
        "blurb": "High and wide all game: progressive receptions, attacking-third volume, and defensive work on the flank.",
        "high": ["prg_r_p90", "padj_tkl", "touch_att3_share", "prg_c_p90", "tkl_att_share"],
        "low": ["touch_def_pen_share"],
    },
    {
        "name": "Wide creator",
        "blurb": "Supplies from the wing: key passes, xAG, and progressive receptions.",
        "high": ["kp_p90", "xag_p90", "ppa_p90", "prg_r_p90", "ast_p90"],
        "low": ["padj_clr"],
    },
    {
        "name": "Inside forward",
        "blurb": "Cuts inside to finish: xG, shots, and penalty-box touches rather than hold-up play.",
        "high": ["npxg_p90", "sh_p90", "touch_att_pen_share", "prg_c_p90", "cpa_p90"],
        "low": ["aerial_p90", "padj_tkl"],
    },
    {
        "name": "Winger",
        "blurb": "Stretches the pitch: take-ons, progressive carries, and attacking-third volume.",
        "high": ["take_att_p90", "prg_c_p90", "touch_att3_share", "carries_p90", "takeons_per_touch"],
        "low": ["padj_clr"],
    },
    {
        "name": "Target striker",
        "blurb": "Holds the ball up and wins aerials in the box; finishing volume over chance creation.",
        "high": ["aerial_p90", "touch_att_pen_share", "npxg_p90", "sh_p90", "aerial_win_share"],
        "low": ["prg_p_p90", "take_att_p90"],
    },
    {
        "name": "Poacher",
        "blurb": "Lives in the six-yard box: shots and xG, almost no defensive or build-up work.",
        "high": ["npxg_p90", "sh_p90", "touch_att_pen_share", "gls_p90", "sot_p90"],
        "low": ["padj_tkl", "prg_p_p90", "touch_mid3_share"],
    },
    {
        "name": "False nine",
        "blurb": "Drops off the front to combine: key passes and xAG with some finishing threat.",
        "high": ["kp_p90", "xag_p90", "ppa_p90", "prg_r_p90", "pass_att_p90"],
        "low": ["padj_clr", "aerial_p90"],
    },
    {
        "name": "Shadow striker",
        "blurb": "Arrives from midfield: penalty-box touches, shots, and progressive receptions.",
        "high": ["touch_att_pen_share", "sh_p90", "prg_r_p90", "npxg_p90", "cpa_p90"],
        "low": ["padj_clr", "padj_blocks"],
    },
    {
        "name": "Pressing forward",
        "blurb": "Leads the press from the front: tackles and interceptions plus attacking-third work.",
        "high": ["padj_tkl", "padj_int", "touch_att3_share", "sh_p90", "tkl_att_share"],
        "low": ["padj_clr"],
    },
]

KEEPER_CATALOG: list[dict] = [
    {
        "name": "Sweeper-keeper",
        "blurb": "Steps out of the box: sweeper actions and a high starting position.",
        "high": ["gk_opa_p90", "gk_sweeper_avg_dist"],
        "low": ["gk_launch_pct"],
    },
    {
        "name": "Shot-stopper",
        "blurb": "Lives on the line: save percentage and PSxG overperformance.",
        "high": ["gk_save_share", "gk_psxg_plus_minus_p90", "gk_pka_save_share"],
        "low": ["gk_opa_p90"],
    },
    {
        "name": "Distributor",
        "blurb": "Starts attacks with the feet: long launches and long goal-kicks.",
        "high": ["gk_launch_pct", "gk_avg_pass_len", "gk_goal_kick_launch_pct"],
        "low": ["gk_throws_share"],
    },
]


@dataclass(frozen=True)
class ClusterFit:
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    labels: list[str]
    blurbs: list[str]
    n_components: int
    feature_names: list[str]


def _feat_index(names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}


def _prototype_vectors(names: list[str], catalog: list[dict]) -> np.ndarray:
    idx = _feat_index(names)
    proto = np.zeros((len(catalog), len(names)), dtype=np.float64)
    for i, spec in enumerate(catalog):
        for name in spec["high"]:
            if name in idx:
                proto[i, idx[name]] = 1.35
        for name in spec["low"]:
            if name in idx:
                proto[i, idx[name]] = -0.85
    return proto


def _assign_catalog(means: np.ndarray, names: list[str]) -> tuple[list[str], list[str]]:
    proto = _prototype_vectors(names, ARCHETYPE_CATALOG)
    k = means.shape[0]
    cost = np.zeros((k, len(ARCHETYPE_CATALOG)), dtype=np.float64)
    for i in range(k):
        for j in range(len(ARCHETYPE_CATALOG)):
            cost[i, j] = np.linalg.norm(means[i] - proto[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    labels = ["Unlabelled style"] * k
    blurbs = [""] * k
    for r, c in zip(row_ind, col_ind):
        labels[int(r)] = ARCHETYPE_CATALOG[int(c)]["name"]
        blurbs[int(r)] = ARCHETYPE_CATALOG[int(c)]["blurb"]
    return labels, blurbs


def fit_gmm(X: np.ndarray, seed: int = 0, n_components: int = N_ARCHETYPES) -> ClusterFit:
    Xz = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    k = min(n_components, len(ARCHETYPE_CATALOG), max(2, Xz.shape[0] // 8))
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        n_init=4,
        max_iter=200,
        random_state=seed,
    )
    gmm.fit(Xz)
    labels, blurbs = _assign_catalog(gmm.means_, OUTFIELD_FEATURES)
    return ClusterFit(
        means=gmm.means_.astype(np.float32),
        covariances=gmm.covariances_.astype(np.float32),
        weights=gmm.weights_.astype(np.float32),
        labels=labels,
        blurbs=blurbs,
        n_components=k,
        feature_names=list(OUTFIELD_FEATURES),
    )


def save_gmm(fit: ClusterFit, path: Path | None = None) -> Path:
    dest = path or GMM_NPZ
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dest,
        means=fit.means,
        covariances=fit.covariances,
        vars=fit.covariances,
        weights=fit.weights,
        labels=np.array(fit.labels),
        names=np.array(fit.labels),
        blurbs=np.array(fit.blurbs),
        n_components=np.array(fit.n_components),
        k=np.array(fit.n_components),
        feature_names=np.array(fit.feature_names),
        groups=np.array(list(FEATURE_GROUPS.keys())),
    )
    return dest


def load_gmm(path: Path | None = None) -> ClusterFit:
    src = path or GMM_NPZ
    z = np.load(src, allow_pickle=True)
    files = set(z.files)
    if "n_components" in files:
        n = int(z["n_components"])
    else:
        n = int(z["k"])
    if "covariances" in files:
        cov = z["covariances"]
    else:
        cov = z["vars"]
    if "labels" in files:
        labels = [str(x) for x in z["labels"].tolist()]
    else:
        labels = [str(x) for x in z["names"].tolist()]
    if "blurbs" in files:
        blurbs = [str(x) for x in z["blurbs"].tolist()]
    else:
        blurbs = [""] * n
    if "feature_names" in files:
        feature_names = [str(x) for x in z["feature_names"].tolist()]
    else:
        feature_names = list(OUTFIELD_FEATURES)
    return ClusterFit(
        means=z["means"],
        covariances=cov,
        weights=z["weights"],
        labels=labels,
        blurbs=blurbs,
        n_components=n,
        feature_names=feature_names,
    )


def responsibilities(fit: ClusterFit, x_z: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x_z, dtype=np.float64).reshape(-1), nan=0.0)
    d = fit.means.shape[1]
    x = x[:d]
    var = np.asarray(fit.covariances, dtype=np.float64) + 1e-6
    diff = x[None, :] - np.asarray(fit.means, dtype=np.float64)
    log_det = np.sum(np.log(var), axis=1)
    quad = np.sum(diff * diff / var, axis=1)
    log_p = np.log(np.asarray(fit.weights, dtype=np.float64) + 1e-12) - 0.5 * (
        log_det + quad + d * np.log(2 * np.pi)
    )
    log_p -= log_p.max()
    p = np.exp(log_p)
    return p / p.sum()


def prototype_scores(x_z: np.ndarray, names: list[str] | None = None) -> np.ndarray:
    """Cosine match of a z-vector against the named style prototypes.

    GMM posteriors in 40+ dimensions collapse to one-hot; this is what we show.
    """
    names = names or list(OUTFIELD_FEATURES)
    proto = _prototype_vectors(names, ARCHETYPE_CATALOG)
    x = np.nan_to_num(np.asarray(x_z, dtype=np.float64).reshape(-1)[: len(names)], nan=0.0)
    xn = x / (np.linalg.norm(x) + 1e-8)
    pn = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    return pn @ xn


def top_archetypes(fit: ClusterFit | None, x_z: np.ndarray, n: int = 3) -> list[dict]:
    names = list(fit.feature_names) if fit is not None else list(OUTFIELD_FEATURES)
    scores = prototype_scores(x_z, names)
    tau = 0.12
    z = scores / tau
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    order = np.argsort(-p)
    out = []
    for k in order[:n]:
        if p[k] < 0.08:
            continue
        spec = ARCHETYPE_CATALOG[int(k)]
        out.append({"name": spec["name"], "blurb": spec["blurb"], "weight": float(p[k])})
    return out


def responsibilities_for(fit: ClusterFit, x_z: np.ndarray) -> list[dict]:
    return top_archetypes(fit, x_z)


def keeper_archetypes(x_z: np.ndarray, n: int = 2) -> list[dict]:
    idx = _feat_index(GK_FEATURES)
    x = np.nan_to_num(np.asarray(x_z, dtype=np.float64).reshape(-1), nan=0.0)
    scores = []
    for spec in KEEPER_CATALOG:
        high = [x[idx[name]] for name in spec["high"] if name in idx]
        low = [x[idx[name]] for name in spec["low"] if name in idx]
        score = (float(np.mean(high)) if high else 0.0) - 0.35 * (float(np.mean(low)) if low else 0.0)
        scores.append(score)
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr - arr.max()
    p = np.exp(arr)
    p = p / p.sum()
    order = np.argsort(-p)
    out = []
    for i in order[:n]:
        if p[i] < 0.12:
            continue
        spec = KEEPER_CATALOG[int(i)]
        out.append({"name": spec["name"], "blurb": spec["blurb"], "weight": float(p[i])})
    return out


def pca2(embeddings: np.ndarray) -> np.ndarray:
    x = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def run() -> None:
    import pandas as pd

    from kindred.similarity import build_index

    features = pd.read_parquet(FEATURES_PARQUET)
    index = build_index(features, "outfield")
    fit = fit_gmm(index.features)
    save_gmm(fit)

    if EMBEDDINGS_NPZ.exists():
        blob = np.load(EMBEDDINGS_NPZ, allow_pickle=False)
        xy_out = pca2(blob["outfield"])
        xy_gk = pca2(blob["keeper"]) if "keeper" in blob.files else np.zeros((0, 2))
        np.savez(
            PROJECTION_NPZ,
            outfield_xy=xy_out,
            outfield_ids=blob["outfield_ids"],
            keeper_xy=xy_gk,
            keeper_ids=blob["keeper_ids"] if "keeper_ids" in blob.files else np.array([], dtype="U"),
        )
    print(f"GMM k={fit.n_components} archetypes={fit.labels}")
    print(f"wrote {GMM_NPZ}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit Pip Guardiola named archetypes")
    parser.parse_args(argv)
    run()


if __name__ == "__main__":
    main()
