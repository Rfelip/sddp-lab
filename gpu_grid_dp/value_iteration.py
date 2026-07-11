"""Cyclic (infinite-horizon, discounted) value iteration to a fixed point.

Maintains one value function per month V_t (t = 0..11), each a flat array over
the S^d state grid. A sweep updates every month by Gauss-Seidel backward
induction on the cyclic graph (V_t <- T_t V_{(t+1) mod 12}); the discount
gamma < 1 makes the cyclic operator a contraction, so the sweep converges to
the unique periodic value functions (~65 sweeps per the lit-review verdict).

Inflows are fixed SAA samples (drawn once with `cfg.seed`), so the fixed point
is well defined. The policy simulator draws FRESH (out-of-sample) inflows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .bellman import build_operator
from .grid import build_action_grid


# --------------------------------------------------------------------------- #
# inflow SAA samples
# --------------------------------------------------------------------------- #
def sample_inflows(spec, cfg, rng=None):
    """[n_seasons, n_res, R] lognormal inflow samples (fixed SAA scenario set)."""
    rng = rng or np.random.default_rng(cfg.seed)
    T, d, R = cfg.n_stages_cycle, spec.n_res, cfg.n_inflow_samples
    out = np.zeros((T, d, R))
    for t in range(T):
        for i in range(d):
            mu, sig = spec.inflow_mu[i, t % spec.n_seasons], spec.inflow_sigma[i, t % spec.n_seasons]
            out[t, i] = rng.lognormal(mean=mu, sigma=sig, size=R)
    return out


@dataclass
class VIResult:
    V: np.ndarray            # [n_seasons, n_states]
    sweeps: int
    converged: bool
    history: list            # per-sweep relative delta
    statics: dict
    seconds: float


def run_value_iteration(spec, cfg, verbose=True) -> VIResult:
    import jax.numpy as jnp

    apply_stage, statics = build_operator(spec, cfg)
    n_states = statics["n_states"]
    T = cfg.n_stages_cycle

    inflows = sample_inflows(spec, cfg)                       # [T,d,R]
    inflows_j = [jnp.asarray(inflows[t], jnp.float32) for t in range(T)]
    # traced jnp scalars -> no per-load jit recompile across the 12 months
    loads = [jnp.asarray(spec.load_cycle[t % len(spec.load_cycle)], jnp.float32)
             for t in range(T)]

    V = [jnp.zeros(n_states, jnp.float32) for _ in range(T)]
    history = []
    t0 = time.time()
    converged = False
    sweep = 0
    for sweep in range(1, cfg.vi_max_sweeps + 1):
        delta = 0.0
        scale = 1e-6
        for t in range(T - 1, -1, -1):                       # backward Gauss-Seidel sweep
            v_new = apply_stage(V[(t + 1) % T], loads[t], inflows_j[t])
            delta = max(delta, float(jnp.max(jnp.abs(v_new - V[t]))))
            scale = max(scale, float(jnp.max(jnp.abs(v_new))))
            V[t] = v_new
        rel = delta / scale
        history.append(rel)
        if verbose:
            print(f"sweep {sweep:3d}  rel-delta {rel:.3e}  scale {scale:.3e}")
        if rel < cfg.vi_tol:
            converged = True
            break

    return VIResult(
        V=np.stack([np.asarray(v) for v in V]),
        sweeps=sweep, converged=converged, history=history,
        statics={k: statics[k] for k in ("n_states", "n_actions", "n_chunks")},
        seconds=time.time() - t0,
    )


# --------------------------------------------------------------------------- #
# value lookup + out-of-sample policy simulation (for the SDDP cross-check)
# --------------------------------------------------------------------------- #
def value_at(spec, cfg, res: VIResult, storage, season=0):
    """Multilinear value V_season(storage) at an arbitrary storage vector."""
    import jax.numpy as jnp
    from .grid import make_grid_spec, multilinear_gather
    gs = make_grid_spec(spec, cfg.n_states_per_res)
    v = multilinear_gather(jnp.asarray(res.V[season], jnp.float32),
                           jnp.asarray(storage, jnp.float32)[None, :], gs)
    return float(v[0])


def simulate_policy(spec, cfg, res: VIResult, n_paths=500, n_years=20, seed=12345):
    """Out-of-sample cost of the GREEDY RECOURSE policy under fresh inflows.

    At each step the action is recomputed by a one-step lookahead over the
    action grid against the converged V_{t+1} (the recourse decision adapts to
    the realized inflow — consistent with the Bellman operator). Returns
    dict(mean_discounted, mean_stage_cost, std_stage_cost).
    """
    from .grid import multilinear_gather_np
    rng = np.random.default_rng(seed)
    S, P, d = cfg.n_states_per_res, cfg.n_actions_per_res, spec.n_res
    releases = build_action_grid(spec, P)                    # [A,d]
    hydro_gen = (releases * spec.productivity).sum(axis=-1)  # [A]
    cap = np.asarray(spec.thermal.cap); cost = np.asarray(spec.thermal.cost)
    clo = np.concatenate([[0.0], np.cumsum(cap)[:-1]])
    T = cfg.n_stages_cycle
    gamma = cfg.discount

    def thermal_vec(load, hg):                               # [A] thermal cost per action
        dd = np.clip(load - hg, 0.0, None)[:, None]          # [A,1]
        steps = (cost * np.clip(dd - clo, 0.0, cap)).sum(axis=-1)
        return steps + spec.thermal.deficit_cost * np.clip(load - hg - cap.sum(), 0.0, None)

    total_disc = np.zeros(n_paths)
    total_stage = np.zeros(n_paths)
    n_steps = n_years * T
    for p in range(n_paths):
        s = spec.init_storage.astype(float).copy()
        disc = 0.0
        for k in range(n_steps):
            t = k % T
            infl = np.array([rng.lognormal(spec.inflow_mu[i, t % spec.n_seasons],
                                           spec.inflow_sigma[i, t % spec.n_seasons])
                             for i in range(d)])
            next_raw = s[None, :] + infl[None, :] - releases  # [A,d]
            feas = np.all(next_raw >= spec.min_storage - 1e-3, axis=-1)
            spill = np.clip(next_raw - spec.max_storage, 0.0, None)
            nxt = np.clip(next_raw, spec.min_storage, spec.max_storage)  # [A,d]
            immediate = (thermal_vec(float(spec.load_cycle[t]), hydro_gen)
                         + (spill * spec.spill_penalty).sum(axis=-1))    # [A]
            v_next = multilinear_gather_np(res.V[(t + 1) % T], nxt, spec, S)  # [A]
            q = np.where(feas, immediate + gamma * v_next, np.inf)
            a = int(np.argmin(q))
            realized = float(immediate[a])
            disc += (gamma ** k) * realized
            total_stage[p] += realized
            s = nxt[a]
        total_disc[p] = disc
    return dict(mean_discounted=float(total_disc.mean()),
                mean_stage_cost=float(total_stage.mean() / n_steps),
                std_stage_cost=float((total_stage / n_steps).std()))
