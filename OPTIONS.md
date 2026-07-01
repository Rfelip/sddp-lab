# sddp-lab — Options for each part

The consolidated ("official") lab is **modular by configuration**: each part of a study is a
selectable option, so a run is assembled from independent choices rather than a bespoke script.
This is the menu — the config key that selects each option, its values, and an example that
demonstrates it. Nothing here is a new abstraction; it documents the choices the lab already
dispatches on, now unified on one branch (`integration`: AR-multiplicative + MarkovChain +
saofrancisco + parallel-scheme + the Evaluation module + the profiler).

## 1. Inflow stochastic process — `data/scenarios.jsonc : inflow.stochastic_process.kind`
| Option | What it is | Positivity | Example |
|---|---|---|---|
| `Naive` | stagewise-independent SAA (marginals only) | LogNormal, positive | `example/4ree`, `example/saofrancisco` |
| `AutoRegressive` | AR/PAR state-augmentation (TS-SDDP). **Multiplicative (Shapiro eq 5.7)**: `ε` enters as a per-opening coefficient → stays LP, non-negative; `INFLOW≥0` elastic slack floors residual drought negativity | positive by construction | `example/4ree_ar`, `example/1dsin_ar` |
| `MarkovChain` | MC-SDDP: per-season K-state lattice + KxK transitions on a Markovian policy graph. Fit from a source AR **or directly from historical data** (`historical_file` key) | positive (real/quantized values) | `example/4ree_mc`, `example/1dsin_mc` |

Note: `AutoRegressive` and `MarkovChain` are the two ways to carry inter-stage dependence
(state-augmentation vs discretization); `Naive` ignores it. A genuine continuous-state AR needs
the multiplicative (not log/exp-link) form — the exp-link form is nonlinear/non-LP (SDDiP).

## 2. Scenario graph / horizon — `data/algorithm.jsonc : scenario_graph.kind`
| Option | What it is | Example |
|---|---|---|
| `RegularScenarioGraph` (finite / ExplicitHorizon) | finite T-stage tree | `example/4ree` (12 stages) |
| `CyclicScenarioGraph` | infinite-horizon, `discount_rate` + `cycle_length` + `max_depth` | `example/saofrancisco` (γ=0.9, 12-cycle, 120 depth) |

## 3. Risk measure — `data/tasks.jsonc : Policy.params.risk_measure`
`Expectation` (risk-neutral) · `CVaR` (params `alpha`, `lambda`) — e.g. the bpbp 4ree benchmark uses `CVaR(0.5, 0.5)`.

## 4. Solver — `scripts/run_and_score.jl --solver` (or the optimizer passed to `Entrypoint`)
| Option | When |
|---|---|
| `highs` (default) | robust on degenerate/CVaR LPs; `presolve=off, simplex, threads=1`. **Prefer this** — GLPK can hang mid-simplex on hard CVaR pivots |
| `glpk` | lightweight; fine for easy cases |

## 5. Parallel scheme — `data/tasks.jsonc : Policy.params.parallel_scheme`
`Serial` · `Asynchronous` · `Threaded` (→ `SDDP.Threaded()`, ~6× on the run-03 matrix — 82% of wall is HiGHS LP-solve) — see `src/Tasks/parallelscheme.jl`.

## 6. Evaluation / out-of-sample — `SDDPlab.Evaluation` + `scripts/run_and_score.jl`
| Option | How | Notes |
|---|---|---|
| in-sample | `in_sample_cost(sims)` (InSampleMonteCarlo) | each method on its own trained support — over-optimism diagnostic |
| **fair-OOS on real history** | `--historical <canonical CSV>` → `oos_costs(...)` | all methods scored on the SAME held-out `year,stage,h1..hN` bank (common random numbers) — the fair metric |
| cyclic OOS | `--cyclic --period P` | season-node rollout for infinite-horizon graphs (saofrancisco) |

Process-aware feed is automatic: Naive/MC use the direct `INFLOW==ω` link; AR back-outs the
standardized residual (pass `--ar-params`). Per-basin builders (raw ONS/WSMPI series → the
canonical CSV) live with each example's `data/`.

## 7. Profiling — `scripts/profile_case.jl`
Any case → per-iteration wall + `model.timer_output` forward/backward/calculate_bound breakdown
+ slowdown-vs-cuts. A guarded BPBP-bunching-stats block activates under the `]dev` BPBP build
(see `handoff-bpbp-lab-profiler.md`); inert on the stock lab.

## Single-call assembly
```bash
# train + benchmark two processes on the same benchmark, one command:
julia --project=. scripts/run_and_score.jl \
    --example example/4ree --example example/4ree_mc \
    --iters 300 --solver highs \
    --historical example/4ree_mc/data/historical_scenarios.csv --out cmp.json
```
