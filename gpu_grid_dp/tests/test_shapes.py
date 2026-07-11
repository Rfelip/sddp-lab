"""CPU sanity tests for the grid-DP spike (run with JAX_PLATFORMS=cpu).

These validate the math pieces independently of the GPU so a green run here
means the kernels are correct before spending GPU time:

    JAX_PLATFORMS=cpu ~/Desktop/experimentos/.venv-rocm/bin/python -m pytest \
        gpu_grid_dp/tests/test_shapes.py -q
"""

from __future__ import annotations

import numpy as np


def test_thermal_monotone_convex():
    import jax.numpy as jnp
    from gpu_grid_dp.problem import ThermalStack
    from gpu_grid_dp.thermal import ThermalCost
    stack = ThermalStack(cost=np.array([10.0, 20.0, 50.0]),
                         cap=np.array([100.0, 100.0, 100.0]), deficit_cost=1000.0)
    tc = ThermalCost(stack)
    d = jnp.linspace(0, 400, 50)
    c = np.asarray(tc(d)).astype(np.float64)
    tol = 1e-5 * abs(c).max()                          # float32 noise ~ mag*1e-7
    assert np.all(np.diff(c) >= -tol)                  # nondecreasing
    assert np.all(np.diff(c, 2) >= -tol)               # convex (merit order)
    # first 100 MW at marginal 10 -> cost 1000 at d=100
    assert abs(float(tc(jnp.array(100.0))) - 1000.0) < 1e-2
    # beyond 300 MW total cap -> deficit slope 1000
    assert float(tc(jnp.array(301.0))) - float(tc(jnp.array(300.0))) > 900


def test_cvar_limits():
    import jax.numpy as jnp
    from gpu_grid_dp.cvar import cvar_reduce
    z = jnp.asarray(np.arange(1, 11, dtype=np.float32))[None]  # [1,10]
    assert abs(float(cvar_reduce(z, 0.1, 0.0)[0]) - 5.5) < 1e-4    # lam=0 -> mean
    assert abs(float(cvar_reduce(z, 1.0, 1.0)[0]) - 5.5) < 1e-4    # alpha=1 -> mean
    assert abs(float(cvar_reduce(z, 0.1, 1.0)[0]) - 10.0) < 1e-4   # worst 10% = {10}
    assert abs(float(cvar_reduce(z, 0.2, 1.0)[0]) - 9.5) < 1e-4    # worst 20% = {10,9}


def test_interp_exact_on_nodes():
    import jax.numpy as jnp
    from gpu_grid_dp.grid import make_grid_spec, multilinear_gather
    from gpu_grid_dp.problem import build_2ree_toy
    spec = build_2ree_toy()
    S = 6
    gs = make_grid_spec(spec, S)
    from gpu_grid_dp.grid import build_state_grid
    states, _ = build_state_grid(spec, S)
    V = jnp.asarray(np.arange(S * S, dtype=np.float32))     # 2-res -> S^2 nodes
    got = np.asarray(multilinear_gather(V, jnp.asarray(states, jnp.float32), gs))
    assert np.allclose(got, np.arange(S * S), atol=1e-3)    # exact at grid nodes


def test_one_stage_runs():
    from gpu_grid_dp.config import GridDPConfig
    from gpu_grid_dp.problem import build_2ree_toy
    from gpu_grid_dp.bellman import build_operator
    import jax.numpy as jnp
    spec = build_2ree_toy()
    cfg = GridDPConfig(n_states_per_res=8, n_actions_per_res=5, n_inflow_samples=10)
    apply_stage, st = build_operator(spec, cfg)
    V0 = jnp.zeros(st["n_states"], jnp.float32)
    mean_infl = np.exp(spec.inflow_mu[:, 0])               # [d]
    infl = jnp.asarray(np.broadcast_to(mean_infl[:, None],
                       (spec.n_res, cfg.n_inflow_samples)), jnp.float32)
    v = apply_stage(V0, jnp.float32(spec.load_cycle[0]), infl)
    assert v.shape == (st["n_states"],)
    assert np.all(np.asarray(v) >= 0)                      # costs nonnegative
    # zero-release action is always feasible -> finite value everywhere
    assert np.all(np.isfinite(np.asarray(v)))
