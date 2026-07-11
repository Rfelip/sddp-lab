"""Mean-CVaR risk reduction over the inflow-sample axis.

The risk measure (lab / Avila 2024 convention, run-03):

    rho_lambda[Z] = (1 - lambda) * E[Z] + lambda * AVaR_alpha[Z]

with lambda = 0 -> risk-neutral expectation (matches a risk-neutral SDDP run).
Z is a COST (we minimize), so the dangerous tail is the UPPER alpha-fraction.

AVaR (= CVaR) for R equiprobable samples is the exact empirical average of the
worst alpha-fraction, splitting the boundary sample (Rockafellar-Uryasev):

    sort Z descending; k = alpha * R;
    AVaR = (1/k) [ sum_{i<=floor(k)} z_(i) + (k - floor(k)) * z_(floor(k)+1) ]

Reduces the last axis (samples) of a tensor of any leading shape.
"""

from __future__ import annotations

import math

import jax.numpy as jnp


def cvar_reduce(z, alpha: float, lam: float):
    """rho_lambda over the last axis of `z` ([..., R] -> [...])."""
    mean = z.mean(axis=-1)
    if lam == 0.0:
        return mean

    R = z.shape[-1]
    k = alpha * R
    kf = int(math.floor(k))
    frac = k - kf

    zs = jnp.sort(z, axis=-1)[..., ::-1]            # descending along samples
    # weights: 1 for the first kf worst samples, `frac` for the next, scaled 1/k
    w = jnp.zeros((R,), dtype=z.dtype)
    if kf > 0:
        w = w.at[:kf].set(1.0)
    if kf < R and frac > 0.0:
        w = w.at[kf].set(frac)
    w = w / max(k, 1e-12)
    avar = (zs * w).sum(axis=-1)
    return (1.0 - lam) * mean + lam * avar
