"""Train the contrastive encoder. Consecutive-season pairs are held out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import nnx

from kindred.features import (
    FEATURE_GROUPS,
    GK_FEATURES,
    GK_GROUPS,
    OUTFIELD_FEATURES,
    _gk_style,
    _outfield_style,
)
from kindred.model import EncoderBundle, StyleEncoder, encode_matrix, info_nce, save_encoder
from kindred.paths import EMBEDDINGS_NPZ, ENCODER_NPZ, FEATURE_META_JSON, FEATURES_PARQUET
from kindred.similarity import build_index, weighted_matrix

POISSON_OUTFIELD = [
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
]

POISSON_GK = [
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


def _z_lookup(meta: dict, role: str, years: np.ndarray, comps: np.ndarray, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    params = meta["z_params"][role]
    mu = np.zeros((len(years), len(feats)), dtype=np.float32)
    sd = np.ones((len(years), len(feats)), dtype=np.float32)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, (year, comp) in enumerate(zip(years, comps)):
        key = f"{int(year)}|{comp}"
        if key not in cache:
            block = params.get(key)
            if not block:
                cache[key] = (
                    np.zeros(len(feats), dtype=np.float32),
                    np.ones(len(feats), dtype=np.float32),
                )
            else:
                cache[key] = (
                    np.array([block[f]["mean"] for f in feats], dtype=np.float32),
                    np.array([max(block[f]["std"], 1e-6) for f in feats], dtype=np.float32),
                )
        mu[i], sd[i] = cache[key]
    return mu, sd


def poisson_views(
    frame: pd.DataFrame,
    *,
    role: str,
    meta: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Two Poisson resamples of each player-season, re-z-scored with stored params."""
    feats = OUTFIELD_FEATURES if role == "outfield" else GK_FEATURES
    cols = [c for c in (POISSON_OUTFIELD if role == "outfield" else POISSON_GK) if c in frame.columns]
    lam = np.nan_to_num(
        np.clip(frame[cols].to_numpy(dtype=np.float32), 0, 1e6),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    a_counts = rng.poisson(lam).astype(np.float32)
    b_counts = rng.poisson(lam).astype(np.float32)

    def _rebuild(counts: np.ndarray) -> np.ndarray:
        tmp = frame.copy()
        for feat in feats:
            raw_col = f"raw_{feat}"
            if feat not in tmp.columns and raw_col in tmp.columns:
                tmp[feat] = tmp[raw_col]
        for j, col in enumerate(cols):
            tmp[col] = counts[:, j]
        styled = _outfield_style(tmp) if role == "outfield" else _gk_style(tmp)
        raw = styled[feats].to_numpy(dtype=np.float32)
        return np.nan_to_num(raw, nan=0.0)

    raw_a = _rebuild(a_counts)
    raw_b = _rebuild(b_counts)
    mu, sd = _z_lookup(
        meta,
        "outfield" if role == "outfield" else "keeper",
        frame["season_end_year"].to_numpy(),
        frame["comp"].to_numpy(),
        feats,
    )
    z_a = (raw_a - mu) / sd
    z_b = (raw_b - mu) / sd
    return z_a.astype(np.float32), z_b.astype(np.float32)


def _train_role(
    frame: pd.DataFrame,
    meta: dict,
    role: str,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    lr: float,
    temperature: float,
) -> EncoderBundle:
    feats = OUTFIELD_FEATURES if role == "outfield" else GK_FEATURES
    subset = frame.loc[frame["role"] == role].reset_index(drop=True)
    d_in = len(feats)
    rngs = nnx.Rngs(seed)
    encoder = StyleEncoder(d_in, rngs=rngs, hidden=(128, 64), d_out=32)
    optimizer = nnx.Optimizer(encoder, optax.adamw(lr, weight_decay=1e-4), wrt=nnx.Param)
    np_rng = np.random.default_rng(seed)
    n = len(subset)
    print(f"training {role} encoder on {n} rows, d={d_in}")

    def loss_fn(model, xa, xb):
        za = model(xa)
        zb = model(xb)
        return 0.5 * (info_nce(za, zb, temperature) + info_nce(zb, za, temperature))

    @nnx.jit
    def step(model, opt, xa, xb):
        loss, grads = nnx.value_and_grad(loss_fn)(model, xa, xb)
        opt.update(model, grads)
        return loss

    for epoch in range(1, epochs + 1):
        z_a, z_b = poisson_views(subset, role=role, meta=meta, rng=np_rng)
        perm = np_rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 16:
                continue
            xa = jnp.asarray(z_a[idx])
            xb = jnp.asarray(z_b[idx])
            loss = step(encoder, optimizer, xa, xb)
            losses.append(float(loss))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  {role} epoch {epoch:3d}  loss={np.mean(losses):.4f}")

    return EncoderBundle(
        encoder=encoder,
        d_in=d_in,
        d_out=32,
        hidden=(128, 64),
        temperature=temperature,
        role=role,
    )


def run(
    *,
    epochs: int = 40,
    batch_size: int = 256,
    seed: int = 0,
    lr: float = 3e-4,
    temperature: float = 0.07,
) -> None:
    frame = pd.read_parquet(FEATURES_PARQUET)
    meta = json.loads(FEATURE_META_JSON.read_text())
    outfield = _train_role(
        frame, meta, "outfield",
        epochs=epochs, batch_size=batch_size, seed=seed, lr=lr, temperature=temperature,
    )
    keeper = _train_role(
        frame, meta, "keeper",
        epochs=max(40, epochs), batch_size=min(64, batch_size), seed=seed + 1,
        lr=lr, temperature=0.1,
    )
    # Persist outfield as the primary encoder.npz; keepers nested in embeddings file.
    save_encoder(outfield, ENCODER_NPZ)
    save_encoder(keeper, ENCODER_NPZ.with_name("encoder_gk.npz"))

    idx_out = build_index(frame, "outfield")
    idx_gk = build_index(frame, "keeper")
    emb_out = encode_matrix(outfield, idx_out.features)
    emb_gk = encode_matrix(keeper, idx_gk.features)
    np.savez(
        EMBEDDINGS_NPZ,
        outfield=emb_out,
        outfield_ids=idx_out.ids.astype("U"),
        keeper=emb_gk,
        keeper_ids=idx_gk.ids.astype("U"),
    )
    print(f"wrote {ENCODER_NPZ} and {EMBEDDINGS_NPZ}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Kindred contrastive encoder")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.lr,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
