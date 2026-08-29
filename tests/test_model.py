"""Unit tests for InfoNCE and the encoder round-trip."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from kindred.model import EncoderBundle, StyleEncoder, encode_matrix, info_nce, load_encoder, save_encoder


def test_info_nce_is_lowest_on_matched_pairs():
    z = jnp.eye(4)
    matched = float(info_nce(z, z, 0.1))
    shuffled = float(info_nce(z, z[::-1], 0.1))
    assert matched < shuffled


def test_encoder_save_load_roundtrip(tmp_path):
    rngs = nnx.Rngs(0)
    enc = StyleEncoder(8, rngs=rngs, hidden=(16, 8), d_out=4)
    bundle = EncoderBundle(
        encoder=enc, d_in=8, d_out=4, hidden=(16, 8), temperature=0.07, role="outfield"
    )
    x = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    before = encode_matrix(bundle, x)
    path = tmp_path / "enc.npz"
    save_encoder(bundle, path)
    loaded = load_encoder(path)
    after = encode_matrix(loaded, x)
    np.testing.assert_allclose(before, after, atol=1e-5)
    norms = np.linalg.norm(after, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
