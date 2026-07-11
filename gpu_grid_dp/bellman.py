"""The Bellman operator: one risk-averse stage update, chunked over states.

For a chunk of incoming states s, every action (release vector r) and every
inflow sample a:

    next_raw_i = s_i + a_i - r_i
    feasible   = all_i (next_raw_i >= min_i)          # can't draw below min
    spill_i    = relu(next_raw_i - max_i)             # forced spill over max
    s'_i       = clip(next_raw_i, min_i, max_i)
    immediate  = thermal_cost(action, load_t) + sum_i spill_penalty_i * spill_i
    Q_sample   = feasible ? immediate + gamma * V_next(s')  :  +inf
    Q(s,r)     = rho_alpha,lambda over samples [Q_sample]   # mean-CVaR
    V_t(s)     = min_r Q(s,r);   policy(s) = argmin_r Q(s,r)

This is Lee & Sun Alg. 1 (broadcast cost -> reduce over actions -> expectation
over samples) generalized to d reservoirs, with the expectation replaced by a
coherent risk measure and the time loop made cyclic. The full
[states x actions x samples x d] tensor is never materialised — we map over
state chunks (lax.map, sequential, VRAM-bounded).

v1 models independent reservoirs (4ree has no cascade); `spec.downstream` is
ignored and asserted to be all sinks.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .cvar import cvar_reduce
from .grid import (build_action_grid, build_state_grid, make_grid_spec,
                   multilinear_gather)

INF = jnp.float32(1e12)


def build_operator(spec, cfg, dtype=jnp.float32):
    """Precompute static arrays and return (apply_stage, statics).

    apply_stage(V_next_flat, load_t, inflow_t) -> (V_t_flat, policy_t)
      V_next_flat : [n_states] flat value of the *next* stage
      load_t      : scalar demand for this stage
      inflow_t    : [n_res, R] inflow samples for this stage
    """
    assert np.all(np.asarray(spec.downstream) < 0), "v1: independent reservoirs only"

    S, P, R = cfg.n_states_per_res, cfg.n_actions_per_res, cfg.n_inflow_samples
    d = spec.n_res

    states_np, _ = build_state_grid(spec, S)          # [n_states, d]
    releases_np = build_action_grid(spec, P)          # [n_actions, d]
    n_states = states_np.shape[0]
    n_actions = releases_np.shape[0]

    gs = make_grid_spec(spec, S, dtype)
    mins = jnp.asarray(spec.min_storage, dtype)
    maxs = jnp.asarray(spec.max_storage, dtype)
    spill_pen = jnp.asarray(spec.spill_penalty, dtype)
    prod = jnp.asarray(spec.productivity, dtype)

    releases = jnp.asarray(releases_np, dtype)         # [A, d]
    hydro_gen = (releases * prod).sum(axis=-1)         # [A]  total generation

    from .thermal import ThermalCost
    thermal = ThermalCost(spec.thermal, dtype)

    # pad states so chunks are static-sized
    chunk = int(cfg.state_chunk)
    n_chunks = (n_states + chunk - 1) // chunk
    n_pad = n_chunks * chunk
    pad = n_pad - n_states
    states = jnp.asarray(np.pad(states_np, ((0, pad), (0, 0))), dtype)
    states = states.reshape(n_chunks, chunk, d)

    alpha, lam, gamma = cfg.cvar_alpha, cfg.cvar_lambda, jnp.float32(cfg.discount)

    def chunk_kernel(V_next_flat, thermal_cost_act, inflow_t, state_chunk):
        # state_chunk [c,d]; inflow_t [d,R]; releases [A,d]
        s = state_chunk[:, None, None, :]              # [c,1,1,d]
        a = inflow_t.T[None, None, :, :]               # [1,1,R,d]
        r = releases[None, :, None, :]                 # [1,A,1,d]
        next_raw = s + a - r                           # [c,A,R,d]

        feasible = jnp.all(next_raw >= mins - 1e-3, axis=-1)        # [c,A,R]
        spill = jnp.clip(next_raw - maxs, 0.0, None)               # [c,A,R,d]
        next_state = jnp.clip(next_raw, mins, maxs)                # [c,A,R,d]
        spill_cost = (spill * spill_pen).sum(axis=-1)             # [c,A,R]

        v_next = multilinear_gather(V_next_flat, next_state, gs)   # [c,A,R]

        immediate = thermal_cost_act[None, :, None] + spill_cost   # [c,A,R]
        q_sample = jnp.where(feasible, immediate + gamma * v_next, INF)  # [c,A,R]
        # RECOURSE: release adapts to the realized inflow -> choose the best
        # feasible action *per sample*, THEN risk-reduce over samples. This is
        # Lee-Sun Alg.1's order (min over actions, then expectation), with E
        # replaced by the coherent risk measure rho (= what risk-averse SDDP
        # nests: rho_xi[ min_decisions cost ]).
        z = q_sample.min(axis=1)                                  # [c,R] per-sample optimum
        v = cvar_reduce(z, alpha, lam)                            # [c] risk over samples
        return v

    @jax.jit
    def apply_stage(V_next_flat, load_t, inflow_t):
        thermal_cost_act = thermal.of_hydro(hydro_gen, load_t)    # [A]
        f = functools.partial(chunk_kernel, V_next_flat, thermal_cost_act, inflow_t)
        v = jax.lax.map(f, states)                               # [n_chunks, chunk]
        return v.reshape(-1)[:n_states]

    statics = dict(n_states=n_states, n_actions=n_actions, n_chunks=n_chunks,
                   states=states_np, releases=releases_np, thermal=thermal)
    return apply_stage, statics
