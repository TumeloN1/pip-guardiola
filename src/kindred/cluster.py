"""JAX GMM over the learned embedding space, plus gradient group explanations."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from kindred.model import EncoderBundle
from kindred.paths import EMBEDDINGS_NPZ, GMM_NPZ, PROJECTION_NPZ
from kindred.similarity import MatrixIndex, group_scale_vector

ARCHETYPE_SEEDS = [
    "progressive creator",
    "box-to-box engine",
    "wide carrier",
    "penalty-box striker",
    "target forward",
    "ball-playing centre-back",
    "aggressive full-back",
    "destroyer",
    "deep-lying metronome",
    "pressing forward",
]


def _log_gauss(x: jnp.ndarray, mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    # x: (n, d), mean/var: (k, d) → (n, k)
    x = x[:, None, :]
    mean = mean[None, :, :]
    var = jnp.clip(var[None, :, :], 1e-6)
    return -0.5 * jnp.sum(jnp.log(2 * jnp.pi * var) + (x - mean) ** 2 / var, axis=-1)


def gmm_em(
    x: jnp.ndarray,
    k: int,
    *,
    key: jax.Array,
    n_iter: int = 40,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    n, d = x.shape
    key, sub = jax.random.split(key)
    idx = jax.random.choice(sub, n, (k,), replace=False)
    means = x[idx]
    var = jnp.var(x, axis=0) + 1e-3
    vars_ = jnp.tile(var[None, :], (k, 1))
    weights = jnp.ones((k,)) / k

    def body(state, _):
        means, vars_, weights = state
        log_resp = _log_gauss(x, means, vars_) + jnp.log(weights)[None, :]
        log_resp = log_resp - jax.nn.logsumexp(log_resp, axis=1, keepdims=True)
        resp = jnp.exp(log_resp)
        nk = resp.sum(axis=0) + 1e-8
        weights = nk / n
        means = (resp.T @ x) / nk[:, None]
        diff = x[:, None, :] - means[None, :, :]
        vars_ = jnp.sum(resp[:, :, None] * diff * diff, axis=0) / nk[:, None] + 1e-4
        return (means, vars_, weights), None

    (means, vars_, weights), _ = jax.lax.scan(body, (means, vars_, weights), None, length=n_iter)
    log_resp = _log_gauss(x, means, vars_) + jnp.log(weights)[None, :]
    ll = float(jax.nn.logsumexp(log_resp, axis=1).mean())
    return means, vars_, weights, ll


def bic_score(n: int, d: int, k: int, mean_ll: float) -> float:
    n_params = k * (d + d + 1) - 1
    loglik = mean_ll * n
    return n_params * np.log(n) - 2 * loglik


PRETTY = {
    "npxg_p90": "npxG/90",
    "xag_p90": "xAG/90",
    "xa_p90": "xA/90",
    "gls_p90": "goals/90",
    "ast_p90": "assists/90",
    "sh_p90": "shots/90",
    "kp_p90": "key passes/90",
    "prg_p_p90": "progressive passes",
    "prg_c_p90": "progressive carries",
    "prg_r_p90": "progressive receptions",
    "take_att_p90": "take-ons",
    "padj_tkl": "PAdj tackles",
    "padj_int": "PAdj interceptions",
    "padj_blocks": "PAdj blocks",
    "padj_clr": "PAdj clearances",
    "padj_recov": "PAdj recoveries",
    "aerial_p90": "aerials",
    "touch_att3_share": "attacking-third touches",
    "touch_def3_share": "defensive-third touches",
    "touch_mid3_share": "midfield touches",
    "tkl_att_share": "high tackles",
    "tkl_def_share": "low-block tackles",
    "pass_long_share": "long passing",
    "pass_short_share": "short passing",
    "dist_per_carry": "carry distance",
    "pass_final_third_p90": "final-third passes",
    "pass_med_share": "medium passing",
    "shot_dist": "shot distance",
    "fls_p90": "fouls",
    "touch_att_pen_share": "box touches",
    "touch_def_pen_share": "defensive-box touches",
    "ppa_p90": "passes into the box",
    "sot_p90": "shots on target",
    "takeons_per_touch": "take-ons per touch",
    "prg_pass_share": "progressive pass share",
    "cpa_p90": "carries into the box",
    "dis_p90": "dispossessed",
}


def label_from_members(
    resp: np.ndarray,
    z_features: np.ndarray,
    feature_names: list[str],
) -> list[str]:
    names = []
    for j in range(resp.shape[1]):
        top = np.argsort(-resp[:, j])[: min(50, len(resp))]
        mean = z_features[top].mean(axis=0)
        order = np.argsort(-mean)
        a, b = order[0], order[1]
        names.append(f"{PRETTY.get(feature_names[a], feature_names[a])} · {PRETTY.get(feature_names[b], feature_names[b])}")
    return names


def fit_gmm(
    embeddings: np.ndarray,
    *,
    ks: tuple[int, ...] = (6, 8, 10, 12),
    seed: int = 0,
    feature_names: list[str] | None = None,
) -> dict:
    x = jnp.asarray(embeddings)
    key = jax.random.key(seed)
    best = None
    for k in ks:
        key, sub = jax.random.split(key)
        means, vars_, weights, ll = gmm_em(x, k, key=sub)
        bic = bic_score(x.shape[0], x.shape[1], k, ll)
        rec = {
            "k": k,
            "means": np.asarray(means),
            "vars": np.asarray(vars_),
            "weights": np.asarray(weights),
            "ll": ll,
            "bic": float(bic),
        }
        if best is None or rec["bic"] < best["bic"]:
            best = rec
    assert best is not None
    best["names"] = [ARCHETYPE_SEEDS[i % len(ARCHETYPE_SEEDS)] for i in range(best["k"])]
    log_resp = _log_gauss(x, jnp.asarray(best["means"]), jnp.asarray(best["vars"]))
    log_resp = log_resp + jnp.log(jnp.asarray(best["weights"]))[None, :]
    best["resp"] = np.asarray(jax.nn.softmax(log_resp, axis=1))
    return best


def responsibilities_for(gmm: dict, vector: np.ndarray) -> list[dict]:
    x = jnp.asarray(vector)[None, :]
    log_resp = _log_gauss(x, jnp.asarray(gmm["means"]), jnp.asarray(gmm["vars"]))
    log_resp = log_resp + jnp.log(jnp.asarray(gmm["weights"]))[None, :]
    resp = np.asarray(jax.nn.softmax(log_resp, axis=1)[0])
    order = np.argsort(-resp)
    return [{"name": gmm["names"][i], "weight": float(resp[i])} for i in order if resp[i] > 0.02]


def load_gmm(path: Path | None = None) -> dict:
    src = path or GMM_NPZ
    blob = np.load(src, allow_pickle=True)
    return {
        "k": int(blob["k"]),
        "means": blob["means"],
        "vars": blob["vars"],
        "weights": blob["weights"],
        "names": blob["names"].tolist(),
        "bic": float(blob["bic"]),
    }


def save_gmm(gmm: dict, path: Path | None = None) -> Path:
    dest = path or GMM_NPZ
    np.savez(
        dest,
        k=np.array(gmm["k"]),
        means=gmm["means"],
        vars=gmm["vars"],
        weights=gmm["weights"],
        names=np.array(gmm["names"]),
        bic=np.array(gmm["bic"]),
    )
    return dest


def gradient_group_explanations(
    bundle: EncoderBundle,
    index: MatrixIndex,
    query_id: str,
    candidate_id: str,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    lookup = index.id_to_row()
    scale = group_scale_vector(index, weights)
    mat = index.features * scale[None, :]
    q = jnp.asarray(mat[lookup[query_id]])
    c = jnp.asarray(mat[lookup[candidate_id]])
    encoder = bundle.encoder

    def score(xc):
        zq = encoder(q[None, :])[0]
        zc = encoder(xc[None, :])[0]
        return jnp.dot(zq, zc)

    grad = np.asarray(jax.grad(score)(c))
    c_np = np.asarray(c)
    name_to_i = {n: i for i, n in enumerate(index.feature_names)}
    rows = []
    for group, cols in index.groups.items():
        idx = np.array([name_to_i[col] for col in cols])
        rows.append({
            "group": group,
            "score": float(np.sum(grad[idx] * c_np[idx])),
            "weight": float(scale[idx[0]]) if len(idx) else 1.0,
        })
    rows.sort(key=lambda r: -r["score"])
    return rows


def pca2(embeddings: np.ndarray) -> np.ndarray:
    x = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def run() -> None:
    import pandas as pd

    from kindred.paths import FEATURES_PARQUET
    from kindred.similarity import build_index

    blob = np.load(EMBEDDINGS_NPZ, allow_pickle=False)
    gmm = fit_gmm(blob["outfield"])
    features = pd.read_parquet(FEATURES_PARQUET)
    index = build_index(features, "outfield")
    id_to_i = {str(i): n for n, i in enumerate(index.ids.astype(str))}
    order = [id_to_i[str(i)] for i in blob["outfield_ids"].astype(str)]
    gmm["names"] = label_from_members(gmm["resp"], index.features[order], index.feature_names)
    save_gmm(gmm)
    xy_out = pca2(blob["outfield"])
    xy_gk = pca2(blob["keeper"]) if "keeper" in blob.files else np.zeros((0, 2))
    np.savez(
        PROJECTION_NPZ,
        outfield_xy=xy_out,
        outfield_ids=blob["outfield_ids"],
        keeper_xy=xy_gk,
        keeper_ids=blob["keeper_ids"] if "keeper_ids" in blob.files else np.array([], dtype="U"),
    )
    print(f"GMM k={gmm['k']} BIC={gmm['bic']:.1f} archetypes={gmm['names']}")
    print(f"wrote {GMM_NPZ} and {PROJECTION_NPZ}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit Kindred GMM archetypes")
    parser.parse_args(argv)
    run()


if __name__ == "__main__":
    main()
