# Migrating the mpc-vs-sddp SDDP path into sddp-lab

Status as of 2026-07-11. This tracks the 6-step migration from
`~/Desktop/RESEARCH/Doutorado/SDDP-LOCATIONS.md` — consolidating the SDDP-IH / CVaR-SDDP-IH
comparator that `infraestrutura_cientifica/eval/mpc-vs-sddp` (and the `2026-soc-soledad` paper)
currently import as cached CSVs from `experimentos/hydrothermal-mpc`, so the lab produces them.

**Headline: `_eval`'s `config.yaml` was NOT repointed. Steps 1, 3, 4, 5 are done; step 2 is a
documented decision awaiting Ruan's sign-off; step 6 (cut-over) is deliberately not executed
because it would change what the paper reports.** See "The safety gate" below.

## What the old path actually is (the thing being reproduced)

- `hydrothermal-mpc/reference-split/julia_compare.jl` builds a hand-rolled `SDDP.PolicyGraph`
  (`model.jl::make_build`), `UnicyclicGraph(monthly_rate; 12)` with `monthly_rate=(1/1.12)^(1/12)`,
  `inflow_dist=:empirical` (20 seed-42 LogNormal draws per reservoir·month), **LAB deficit mode**
  (single effective segment 1420.34; the "4 segments" are a Python array-shape artifact, 3 of them
  ub=0), HiGHS **presolve on**. Trains `time_limit=120s`, then forward-simulates on 5000 seeded
  OOS inflow paths (`output/shared_inflows.csv`) via `SDDP.Historical`, exports per-(realisation,
  stage) rows through `_stage_row`. `cvar_sddp_trajectories.csv` is the same with
  `SDDP.EAVaR(lambda=0.5, beta=0.15)`, `iteration_limit=1200`.

## Two barriers to a *bit-for-bit* reproduction (both real, established by reading the code)

**Barrier A — model formulation (turned out to be one config knob).** sddp-lab's subproblem
(`Tasks/model.jl`, `System/*`) is a richer LP than `make_build`: turbined-flow×productivity +
separate spillage + downstream routing + a hydro min-generation slack + per-line
DIRECT/REVERSE exchange with penalties, vs. energy-direct generation + full i×j exchange matrix
+ no min-gen. **But for the 4REE instance these collapse:** productivity=1, `min_generation=0`
(slack inactive), no downstream cascade (`downstream_id` empty), single-segment deficit already
== native lab mode, spillage_penalty=1 in both. The *only* live objective difference is the
exchange penalty — and **`make_build`'s objective ignores the `exchange_penalty` column
entirely** (it reads only capacities). So setting `exchange_penalty=0` in the lab case makes the
two LPs match. That is exactly what `example/4ree_cyclic_5bus/data/lines.csv` does.

**Barrier B — training non-determinism (unfixable, governs everything).** `run_sddp` does **not**
`Random.seed!` before `SDDP.train`, and the SDDP-IH path is **wall-clock** limited
(`time_limit=120s` → machine-dependent iteration count). So the trained policy — hence every
number in the frozen CSVs — is **not reproducible run-to-run even by its own generator.** The OOS
simulation on a *fixed* policy is deterministic, but the policy is not. Consequence: the bench's
"reproduce a frozen experiment bit-for-bit" bar is unmeetable here by any route. The only
meaningful equivalence is **statistical / distributional** over the 5000 OOS realisations.

## Step-by-step status

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1 | Cyclic + 5-bus case | **DONE** | `example/4ree_cyclic_5bus` — net-new. 5-bus SUDESTE/SUL/NORDESTE/NORTE/NOFICT1 exchange + `CyclicScenarioGraph`, discount `(1/1.12)^(1/12)=0.99060`. Loads + trains, 0 numeric issues. |
| 2 | Inflow provenance | **DECISION MADE (see below)** | Recommend adopting the lab's Naive bank as a statistically-equivalent baseline; needs Ruan sign-off because it changes paper numbers. |
| 3 | Deficit segments | **DONE (no work needed)** | Frozen CSVs use LAB mode = single segment 1420.34 = lab native `buses.csv`. The 4-segment shape never affected the reported numbers. |
| 4 | Solver tuning | **DONE** | The presolve crash trap was specific to the 4-segment ub=0 columns. The native single-segment case trains fine with **presolve=off** (the lab's own default) — verified, 0 numeric issues. No special tuning needed. |
| 5 | Per-stage export | **DONE** | `Evaluation.export_stage_trajectories` + `validate/export_trajectories.jl`. Emits the exact `_eval` schema; verified schema-identical to `sddp_trajectories.csv`, inflow round-trips 0/720, water balance holds. |
| 6 | Validate + repoint | **NOT DONE (deliberate)** | Blocked by the safety gate below. Repointing would silently change the paper's SDDP numbers. |

## Step 2 — the load-bearing methodological call

**Recommendation: Branch B — adopt sddp-lab's Naive scenario bank as the baseline, flagged as a
new (statistically equivalent) baseline, not a bit reproduction of the old CSVs.**

Rationale:
- The lab's `inflow_scenarios.jsonc` marginals are **bit-identical** to hydrothermal-mpc's
  `INFLOW_LOGNORMAL_PARAMS`, its copulas are **Gaussian with identity correlation = independence**
  (same as the old path's independent per-reservoir draws), and the case is set to `seed=42`,
  `branchings=20` — matching the old path's distribution, dependence, seed, and count. The two
  banks differ **only in RNG realization** (Julia `rand(MersenneTwister(42), LogNormal, 20)` per
  reservoir vs. the lab's `rand(rng, SklarDist(GaussianCopula(I), marginals))`). They are
  equal-in-distribution; the difference is SAA sampling noise of order 1/√20 — the **same order**
  the old path already injects via Barrier B (its own unseeded retraining).
- Branch A (bit-reproduce the old bank) would require a *new* lab stochastic-process kind that
  ingests a precomputed explicit sample bank (the Naive `__generate_saa` always calls `rand`;
  there is no explicit-bank input path), and *even then* Barrier B leaves the trained policy
  non-reproducible. So Branch A buys matched in-sample noise but not matched numbers — high cost,
  no bit-level payoff.

**Because Branch B changes the numbers the paper reports, it is Ruan's call, not the migration's.**
If he prefers Branch A anyway (e.g. to hold the exact in-sample SAA fixed across a future re-run),
the clean implementation is a `ExplicitSamples`/`Historical`-bank stochastic process in
`src/StochasticProcess/` feeding `__generate_saa` from a CSV — scoped but net-new.

## The safety gate (why `_eval` was not repointed)

Per the task bar ("before repointing `_eval` you must verify numerical equivalence... if you
cannot close that loop, STOP and document"): the loop cannot be closed in a way that leaves the
paper's numbers unchanged, because (A) adopting the lab path is by construction a new baseline
(Branch B) and (B) bit reproduction is impossible (Barrier B). Repointing now would silently
alter a published draft's SDDP-IH / CVaR-SDDP-IH figures. So **nothing downstream was touched** —
`hydrothermal-mpc/reference-split/*` and `eval/mpc-vs-sddp/config.yaml` are exactly as before.

## What remains, to finish the cut-over (for Ruan or a future agent)

1. **Ruan decides step 2** (Branch B new-baseline, or fund Branch A explicit-bank process).
2. **Quantify the baseline shift**: train `4ree_cyclic_5bus` to convergence (risk-neutral for
   SDDP-IH; `risk_measure` → `CVaR` for CVaR-SDDP-IH — first VERIFY the lab's `alpha`/`lambda`
   maps to `EAVaR(lambda=0.5, beta=0.15)`), run `validate/export_trajectories.jl` over the full
   5000-path `shared_inflows.csv`, and compare the OOS cost DISTRIBUTION (mean, CVaR95, per-stage
   se/thermal bands) against the frozen CSVs. That gap = the paper-number delta Ruan is signing
   off on.
3. **If approved**, add SDDP-IH + CVaR-SDDP-IH as lab-run sources and repoint
   `eval/mpc-vs-sddp/config.yaml` from `mode: import` (hydrothermal-mpc) to the lab-produced CSVs,
   re-run `make results && make figures && make paper-figs`, and re-verify the paper text against
   the new numbers.

## How to reproduce the verified pieces

```
cd infraestrutura_cientifica/packages/sddp-lab
# Step 1: build+train the new case
julia --project=. validate/run_sddp_baseline.jl example/4ree_cyclic_5bus 500
# Step 5: export per-stage trajectories in _eval's schema (3-path smoke)
julia --project=. validate/export_trajectories.jl example/4ree_cyclic_5bus \
  ~/Desktop/RESEARCH/experimentos/hydrothermal-mpc/output/shared_inflows.csv \
  /tmp/lab_traj.csv 500 SDDP-IH 3
```
