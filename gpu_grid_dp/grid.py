"""State/action grids and d-dimensional multilinear value interpolation.

State grid : each reservoir storage discretized to S levels in [min_i, max_i];
             Cartesian product -> S^d states, flattened row-major (C order).
Action grid: each reservoir turbined release in [0, max_release_i] at P levels;
             Cartesian product -> P^d actions.

Interpolation: the next storage s' = clip(s + inflow - release, min, max) is
continuous, so V_{t+1}(s') is read by multilinear interpolation over the 2^d
grid corners (the d-D generalization of Lee & Sun Alg. 1, lines 4-7). All
gathers are pure tensor ops on a flat V of length S^d.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------------- #
# grids
# --------------------------------------------------------------------------- #
def _cartesian(levels: list[np.ndarray]) -> np.ndarray:
    """Row-major Cartesian product -> [prod, d] (dim 0 = slowest axis)."""
    mesh = np.meshgrid(*levels, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=-1)


def build_state_grid(spec, S: int):
    """Return (states [n_states, d], per-dim level vectors [d, S])."""
    levels = [np.linspace(spec.min_storage[i], spec.max_storage[i], S)
              for i in range(spec.n_res)]
    states = _cartesian(levels)
    return states, np.stack(levels)


def build_action_grid(spec, P: int):
    """Return releases [n_actions, d] (turbined energy per reservoir)."""
    levels = [np.linspace(0.0, spec.max_release[i], P) for i in range(spec.n_res)]
    return _cartesian(levels)


# --------------------------------------------------------------------------- #
# multilinear interpolation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridSpec:
    """Static description of the flat state grid for interpolation."""
    mins: jnp.ndarray      # [d]
    maxs: jnp.ndarray      # [d]
    steps: jnp.ndarray     # [d]  (max-min)/(S-1)
    S: int
    strides: jnp.ndarray   # [d]  row-major flat strides (S^{d-1-i})
    corners: np.ndarray    # [2^d, d]  binary corner offsets (host/static)
    d: int


def make_grid_spec(spec, S: int, dtype=jnp.float32) -> GridSpec:
    d = spec.n_res
    mins = np.asarray(spec.min_storage, np.float64)
    maxs = np.asarray(spec.max_storage, np.float64)
    steps = (maxs - mins) / (S - 1)
    strides = np.array([S ** (d - 1 - i) for i in range(d)], dtype=np.int64)
    # static (numpy) so the 2^d unroll is concrete inside jit; int32 keeps the
    # flat gather index in int32 (S^d < 2^31) without needing jax x64.
    corners = np.array(list(itertools.product([0, 1], repeat=d)), dtype=np.int32)
    return GridSpec(
        mins=jnp.asarray(mins, dtype), maxs=jnp.asarray(maxs, dtype),
        steps=jnp.asarray(steps, dtype), S=S,
        strides=jnp.asarray(strides, jnp.int32),
        corners=corners, d=d,
    )


def multilinear_gather(V_flat, next_state, gs: GridSpec):
    """Interpolate V_flat (len S^d) at `next_state` [..., d] -> [...].

    Pure tensor ops; unrolls the 2^d corners (16 for d=4) inside jit.
    """
    pos = (next_state - gs.mins) / gs.steps               # [..., d] continuous idx
    pos = jnp.clip(pos, 0.0, gs.S - 1.0)
    lo = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, gs.S - 2)  # [..., d]
    frac = pos - lo                                        # [..., d] in [0,1]

    out = jnp.zeros(next_state.shape[:-1], dtype=V_flat.dtype)
    for b in gs.corners:                                   # [d] of {0,1}
        idx = lo + b                                       # [..., d]
        # weight = prod_i (frac_i if b_i else 1-frac_i)
        w = jnp.prod(jnp.where(b == 1, frac, 1.0 - frac), axis=-1)
        flat = (idx * gs.strides).sum(axis=-1)             # [...]
        out = out + w * V_flat[flat]
    return out


def multilinear_gather_np(V_flat, next_state, spec, S):
    """NumPy multilinear interpolation (host-side, for the policy simulator).

    V_flat: [S^d] numpy; next_state: [..., d] numpy. Returns [...].
    """
    mins = np.asarray(spec.min_storage); maxs = np.asarray(spec.max_storage)
    d = spec.n_res
    steps = (maxs - mins) / (S - 1)
    strides = np.array([S ** (d - 1 - i) for i in range(d)])
    pos = np.clip((next_state - mins) / steps, 0.0, S - 1.0)
    lo = np.clip(np.floor(pos).astype(int), 0, S - 2)
    frac = pos - lo
    out = np.zeros(next_state.shape[:-1])
    for b in itertools.product([0, 1], repeat=d):
        b = np.array(b)
        w = np.prod(np.where(b == 1, frac, 1.0 - frac), axis=-1)
        flat = ((lo + b) * strides).sum(axis=-1)
        out = out + w * V_flat[flat]
    return out
