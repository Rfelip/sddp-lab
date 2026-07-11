"""Merit-order thermal + deficit cost as a 1-D lookup over hydro energy.

This is what replaces the LP. Given a stage demand L and total hydro generation
G, the thermal units must serve the residual d = max(0, L - G). Dispatching the
merit order (cheapest first) makes the cost a convex, increasing,
piecewise-linear function of d:

    cost(d) = sum_k  cost_k * clip(d - Clo_k, 0, cap_k)         (thermal steps)
            + deficit_cost * relu(d - Ctotal)                  (unserved load)

where Clo_k is the cumulative capacity below step k and cap_k its width. The
function is fully vectorized and differentiable, so it evaluates over the whole
action grid (or any tensor) in one shot — no solver, no loop over units.

Hydro released beyond demand (G > L) is wasted water, not negative cost, hence
the clip at d = 0: the DP will simply avoid over-releasing.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .problem import ThermalStack


class ThermalCost:
    """Precomputed merit-order cost curve, callable on jnp tensors of demand."""

    def __init__(self, stack: ThermalStack, dtype=jnp.float32):
        cost = np.asarray(stack.cost, dtype=np.float64)
        cap = np.asarray(stack.cap, dtype=np.float64)
        clo = np.concatenate([[0.0], np.cumsum(cap)[:-1]])  # lower edge of each step
        self.cost = jnp.asarray(cost, dtype=dtype)          # [n_th]
        self.cap = jnp.asarray(cap, dtype=dtype)            # [n_th]
        self.clo = jnp.asarray(clo, dtype=dtype)            # [n_th]
        self.ctotal = float(cap.sum())
        self.deficit_cost = float(stack.deficit_cost)
        self.dtype = dtype

    def __call__(self, demand):
        """cost(d) for residual thermal demand `demand` (any shape)."""
        d = jnp.asarray(demand, dtype=self.dtype)[..., None]          # [..., 1]
        steps = self.cost * jnp.clip(d - self.clo, 0.0, self.cap)     # [..., n_th]
        thermal = steps.sum(axis=-1)
        deficit = self.deficit_cost * jnp.clip(
            jnp.asarray(demand, dtype=self.dtype) - self.ctotal, 0.0, None)
        return thermal + deficit

    def of_hydro(self, hydro_gen, load):
        """Stage cost from total hydro generation `hydro_gen` at demand `load`."""
        residual = jnp.clip(jnp.asarray(load, self.dtype)
                            - jnp.asarray(hydro_gen, self.dtype), 0.0, None)
        return self(residual)
