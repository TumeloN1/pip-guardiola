"""Flax NNX encoder + InfoNCE contrastive metric.

Positives are two Poisson resamples of the same player-season (counting stats
are approximately Poisson). The encoder learns which feature gaps are sampling
noise and which are playstyle signal. Consecutive-season positives are held
out of training so adjacent-season MRR stays an honest eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from kindred.paths import ENCODER_NPZ


class StyleEncoder(nnx.Module):
    def __init__(self, d_in: int, *, rngs: nnx.Rngs, hidden: tuple[int, int] = (128, 64), d_out: int = 32):
        h1, h2 = hidden
        self.l1 = nnx.Linear(d_in, h1, rngs=rngs)
        self.l2 = nnx.Linear(h1, h2, rngs=rngs)
        self.l3 = nnx.Linear(h2, d_out, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = jax.nn.relu(self.l1(x))
        h = jax.nn.relu(self.l2(h))
        z = self.l3(h)
        return z / jnp.clip(jnp.linalg.norm(z, axis=-1, keepdims=True), 1e-8)


def info_nce(z_a: jnp.ndarray, z_b: jnp.ndarray, temperature: float) -> jnp.ndarray:
    logits = z_a @ z_b.T / temperature
    labels = jnp.arange(z_a.shape[0])
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


@dataclass
class EncoderBundle:
    encoder: StyleEncoder
    d_in: int
    d_out: int
    hidden: tuple[int, int]
    temperature: float
    role: str


def encode_matrix(bundle: EncoderBundle, matrix: np.ndarray, batch: int = 1024) -> np.ndarray:
    """Encode a (n, d) float32 matrix with the trained encoder."""
    param_graph, state = nnx.split(bundle.encoder)

    @jax.jit
    def _fwd(graph, st, x):
        model = nnx.merge(graph, st)
        return model(x)

    chunks = []
    x = np.asarray(matrix, dtype=np.float32)
    for start in range(0, len(x), batch):
        piece = jnp.asarray(x[start : start + batch])
        chunks.append(np.asarray(_fwd(param_graph, state, piece)))
    return np.concatenate(chunks, axis=0)


def _layer_dump(layer: nnx.Linear) -> dict[str, np.ndarray]:
    kernel = np.asarray(getattr(layer.kernel, "value", layer.kernel))
    bias = np.asarray(getattr(layer.bias, "value", layer.bias))
    return {"kernel": kernel, "bias": bias}


def _layer_load(layer: nnx.Linear, kernel: np.ndarray, bias: np.ndarray) -> None:
    if hasattr(layer.kernel, "value"):
        layer.kernel.value = jnp.asarray(kernel)
        layer.bias.value = jnp.asarray(bias)
    else:
        layer.kernel = jnp.asarray(kernel)
        layer.bias = jnp.asarray(bias)


def save_encoder(bundle: EncoderBundle, path: Path | None = None) -> Path:
    dest = path or ENCODER_NPZ
    enc = bundle.encoder
    l1, l2, l3 = _layer_dump(enc.l1), _layer_dump(enc.l2), _layer_dump(enc.l3)
    np.savez(
        dest,
        d_in=np.array(bundle.d_in),
        d_out=np.array(bundle.d_out),
        hidden=np.array(bundle.hidden),
        temperature=np.array(bundle.temperature),
        role=np.array(bundle.role),
        l1_kernel=l1["kernel"],
        l1_bias=l1["bias"],
        l2_kernel=l2["kernel"],
        l2_bias=l2["bias"],
        l3_kernel=l3["kernel"],
        l3_bias=l3["bias"],
    )
    return dest


def load_encoder(path: Path | None = None) -> EncoderBundle:
    src = path or ENCODER_NPZ
    blob = np.load(src, allow_pickle=False)
    d_in = int(blob["d_in"])
    d_out = int(blob["d_out"])
    hidden = (int(blob["hidden"][0]), int(blob["hidden"][1]))
    temperature = float(blob["temperature"])
    role = str(blob["role"])
    encoder = StyleEncoder(d_in, rngs=nnx.Rngs(0), hidden=hidden, d_out=d_out)
    _layer_load(encoder.l1, blob["l1_kernel"], blob["l1_bias"])
    _layer_load(encoder.l2, blob["l2_kernel"], blob["l2_bias"])
    _layer_load(encoder.l3, blob["l3_kernel"], blob["l3_bias"])
    return EncoderBundle(
        encoder=encoder,
        d_in=d_in,
        d_out=d_out,
        hidden=hidden,
        temperature=temperature,
        role=role,
    )
