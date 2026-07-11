# GPU grid-DP hydrothermal spike

LP-free, GPU-friendly discretized **value-function dynamic programming** as a
cross-check / alternative to SDDP for the 4-reservoir hydrothermal problem.
Method template: **Lee & Sun 2025, "GPU-Accelerated DP for Multistage
Stochastic Energy Storage Arbitrage"** (arXiv:2511.15629), generalized from
1-D storage to **d reservoirs**, with **cyclic** (infinite-horizon, discounted)
value iteration and **mean-CVaR** risk.

This lives inside a **fork** of sddp-lab (branch `gpu-grid-dp-spike`) so the
SDDP.jl baseline and the grid-DP read one identical instance and your main repo
+ running job are untouched.

## The idea (no LP)
Lee–Sun Algorithm 1 is `broadcast cost → max/min over actions → expectation
over samples`, only the time loop sequential. Generalized here:

```
next_raw = s + inflow - release                       # [states, actions, samples, d]
feasible = all(next_raw >= min)                        # can't draw below min
spill    = relu(next_raw - max);  s' = clip(next_raw, min, max)
immediate = thermal_merit_cost(load - Σ release) + Σ spill_penalty·spill   # 1-D lookup, NO LP
Q_sample = feasible ? immediate + γ·V_next(s')  : +∞   # multilinear interp of V on the grid
Q(s,a)   = ρ_{α,λ}[Q_sample]   over samples            # (1-λ)E + λ·AVaR_α  (mean-CVaR)
V_t(s)   = min_a Q(s,a)                                # cyclic: t+1 wraps 12→1
```

Cyclic value iteration sweeps `V_t ← T_t V_{t+1}` to a fixed point (γ=0.9 makes
it a contraction; ~65 sweeps expected). The full `S^d × P^d × R` tensor is
**never materialised** — we map over **state chunks** (`lax.map`, VRAM-bounded).

## Layout
| file | role |
|---|---|
| `config.py` | `GridDPConfig` — S, P, R, γ, α, λ, chunk, case |
| `problem.py` | `ProblemSpec` + lab-CSV loaders + the explicit **2-res toy** |
| `thermal.py` | merit-order 1-D thermal+deficit cost (the LP replacement) |
| `cvar.py` | mean-CVaR sample reduction |
| `grid.py` | state/action grids + d-D multilinear interpolation gather |
| `bellman.py` | the chunked stage operator (Lee–Sun Alg.1 generalized) |
| `value_iteration.py` | cyclic fixed-point sweep + OOS policy simulator |
| `run_spike.py` | CLI: `--dry-run` / `--smoke` / full run |
| `export_lab_case.py` | emits the matching SDDP.jl 2-res cyclic baseline |
| `tests/test_shapes.py` | CPU math tests (thermal, cvar, interp, one stage) |

## Running (GATED — wait for go; needs GPU/threads)
ROCm JAX venv: `~/Desktop/experimentos/.venv-rocm/bin/python` (jax 0.10 + rocm7 plugin).

```bash
PY=~/Desktop/experimentos/.venv-rocm/bin/python
cd ~/Desktop/Doutorado/sddp-lab-gpudp

# 0) safe anytime (no kernel): instance + VRAM estimate
$PY -m gpu_grid_dp.run_spike --case 2ree --dry-run
$PY -m gpu_grid_dp.run_spike --case 4ree --S 15 --P 5 --dry-run

# 1) CPU math tests (no GPU)
JAX_PLATFORMS=cpu $PY -m pytest gpu_grid_dp/tests -q

# 2) CPU smoke (tiny 2-res VI end-to-end)
JAX_PLATFORMS=cpu $PY -m gpu_grid_dp.run_spike --case 2ree --smoke --simulate

# 3) GPU 2-res toy (risk-neutral) + SDDP baseline + compare
$PY -m gpu_grid_dp.run_spike --case 2ree --S 25 --simulate
$PY -m gpu_grid_dp.export_lab_case expectation
julia --project=. --threads 12 src/main.jl example/2ree_cyclic/main.jsonc   # SDDP baseline
$PY validate/compare.py --griddp results_2ree_S25_lam0.0.json --sddp-bound <B>

# 4) scale sweep (4-res): VRAM + time-to-converge wall, gap vs SDDP
for S in 10 15 20 25; do $PY -m gpu_grid_dp.run_spike --case 4ree --S $S --P 5; done
```

## Decision gate
4-res at S≥15 converging in ~minutes within 16 GB **and** matching SDDP within a
few % ⇒ live tool. Can't clear S≈10 or gap >5–10% ⇒ grid-DP dead here →
redirect GPU effort to the block-batch SDDP lane.

## v1 scope / known edges
- **Independent reservoirs** (4ree has no cascade). `downstream` routing is a
  TODO; asserted all-sinks for now.
- PAR(p) as grid axes **kills it** (state blow-up) — hydrology stays a small
  per-month inflow law, never fine grid dims (per the lit-review verdict).
- Coarse grids won't beat SDDP on accuracy; the win is **LP-free GPU** + an
  independent cross-check.
- Untested against a live run (built under a no-run constraint); the gated
  steps above run in order, cheapest first, so failures surface early.
