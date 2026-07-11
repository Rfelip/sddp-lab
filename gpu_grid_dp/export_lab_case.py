"""Generate the SDDP.jl validation baseline for the 2-reservoir toy.

Emits a single-bus, 2-hydro, cyclic, risk-capable sddp-lab case under
example/2ree_cyclic/ whose economics MATCH the grid-DP `ProblemSpec` exactly
(same storage/turbine caps, same merit-order thermals, same demand, same
LogNormal inflow laws). Both solvers then see one instance; the only difference
is grid-DP discretizes where SDDP.jl uses LP/cuts -> that delta is the
discretization gap the spike measures.

Single bus: collapse to one demand node so the lab's network LP reduces to the
grid-DP's single-energy merit-order economics (no lines, no exchange).

Run (pure file I/O, no solve -> safe anytime):
    python -m gpu_grid_dp.export_lab_case
Then the gated SDDP solve:
    julia --project=. --threads 12 src/main.jl example/2ree_cyclic/main.jsonc
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .problem import LAB_ROOT, _read_lab_csv, build_2ree_toy

CASE = LAB_ROOT / "example" / "2ree_cyclic"
N_STAGES = 120          # cyclic unroll depth (10 years monthly)
CYCLE = 12


def _csv(path: Path, header: list[str], rows: list[list]):
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    path.write_text("\n".join(lines) + "\n")


def export(case: str = "2ree", risk: str = "expectation", branchings: int = 30, seed: int = 42):
    from .problem import build_problem
    if case == "2ree":
        spec, CASE = build_2ree_toy(iid=True), LAB_ROOT / "example" / "2ree_cyclic"
    elif case == "4ree":
        # single-bus collapse of 4ree (copper-plate) matching the grid-DP 4ree
        spec, CASE = build_problem("4ree", False), LAB_ROOT / "example" / "4ree_singlebus_cyclic"
    else:
        raise ValueError(case)
    data = CASE / "data"
    data.mkdir(parents=True, exist_ok=True)

    # start from the 4ree template so all boilerplate files exist & validate
    src = LAB_ROOT / "example" / "4ree" / "data"
    for f in ["system.jsonc", "constraints.jsonc"]:
        shutil.copy(src / f, data / f)
    shutil.copy(LAB_ROOT / "example" / "4ree" / "main.jsonc", CASE / "main.jsonc")

    # --- single bus, no lines ------------------------------------------------
    _csv(data / "buses.csv", ["id", "name", "deficit_cost"],
         [[1, '"BUS1"', spec.thermal.deficit_cost]])
    _csv(data / "lines.csv",
         ["id", "name", "source_bus_id", "target_bus_id", "capacity", "exchange_penalty"], [])

    # --- 2 hydros on bus 1 ---------------------------------------------------
    hrows = []
    for i in range(spec.n_res):
        hrows.append([i + 1, f'"{i+1}"', "-", 1, spec.productivity[i],
                      spec.init_storage[i], spec.min_storage[i], spec.max_storage[i],
                      0.0, spec.max_release[i], int(spec.spill_penalty[i])])
    _csv(data / "hydros.csv",
         ["id", "name", "downstream_id", "bus_id", "productivity", "initial_storage",
          "min_storage", "max_storage", "min_generation", "max_generation",
          "spillage_penalty"], hrows)

    # --- thermals: the exact scaled bus-1 merit stack from the spec ----------
    trows = [[k + 1, k + 1, 1, 0.0, float(spec.thermal.cap[k]), float(spec.thermal.cost[k])]
             for k in range(len(spec.thermal.cap))]
    _csv(data / "thermals.csv",
         ["id", "name", "bus_id", "min_generation", "max_generation", "cost"], trows)

    # --- load: bus 1, 120 monthly stages from the 12-month cycle -------------
    lrows = [[1, s, float(spec.load_cycle[(s - 1) % CYCLE])] for s in range(1, N_STAGES + 1)]
    _csv(data / "load.csv", ["bus_id", "stage_index", "value"], lrows)

    # --- stages: 120 monthly ------------------------------------------------
    srows = []
    y, m = 2015, 1
    for s in range(1, N_STAGES + 1):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        srows.append([s, f"{y}-{m:02d}-01", f"{ny}-{nm:02d}-01"])
        y, m = ny, nm
    _csv(data / "stages.csv", ["index", "start_date", "end_date"], srows)

    # --- inflow_scenarios.jsonc: 2 LogNormal marginals, 12 seasons each ------
    marginals = []
    for i in range(spec.n_res):
        dists = [{"season": t + 1, "kind": "LogNormal",
                  "parameters": [float(spec.inflow_mu[i, t]), float(spec.inflow_sigma[i, t])]}
                 for t in range(CYCLE)]
        marginals.append({"id": i + 1, "distributions": dists})
    # independent reservoirs -> identity GaussianCopula per season (matches the
    # grid-DP, which samples each reservoir independently)
    eye = [[1.0 if a == b else 0.0 for b in range(spec.n_res)] for a in range(spec.n_res)]
    copulas = [{"season": t + 1, "kind": "GaussianCopula", "parameters": eye}
               for t in range(CYCLE)]
    (data / "inflow_scenarios.jsonc").write_text(
        json.dumps({"marginal_models": marginals, "copulas": copulas}, indent=2))

    # --- scenarios.jsonc -----------------------------------------------------
    (data / "scenarios.jsonc").write_text(json.dumps({
        "seed": seed, "initial_season": 1, "branchings": branchings,
        "inflow": {"stochastic_process": {"kind": "Naive",
                   "params": {"file": "inflow_scenarios.jsonc"}}},
        "load": {"kind": "DeterministicLoad", "params": {"file": "load.csv"}},
    }, indent=2))

    # --- algorithm.jsonc: cyclic discount 0.9 -------------------------------
    (CASE / "data" / "algorithm.jsonc").write_text(json.dumps({
        "scenario_graph": {"kind": "CyclicScenarioGraph",
            "params": {"discount_rate": 0.9, "cycle_length": CYCLE,
                       "cycle_stage": 1, "max_depth": N_STAGES}},
        "horizon": {"kind": "ExplicitHorizon", "params": {"file": "stages.csv"}},
    }, indent=2))

    # --- tasks.jsonc: policy (Threaded) + simulation -------------------------
    if risk == "cvar":
        risk_block = {"kind": "CVaR", "params": {"alpha": 0.1, "lambda": 0.15}}
    else:
        risk_block = {"kind": "Expectation", "params": {}}
    threaded = {"kind": "Threaded", "params": {}}
    (data / "tasks.jsonc").write_text(json.dumps({"tasks": [
        {"kind": "Policy", "params": {
            "convergence": {"min_iterations": 10, "max_iterations": 1024,
                "stopping_criteria": {"kind": "IterationLimit",
                    "params": {"num_iterations": 500}}},
            "risk_measure": risk_block,
            "parallel_scheme": threaded,
            "results": {"path": "out/policy", "save": True,
                        "format": {"kind": "ParquetFormat", "params": {}}}}},
        {"kind": "Simulation", "params": {
            "num_simulated_series": 2000,
            "policy": {"load": False, "path": "out/policy",
                       "format": {"kind": "ParquetFormat", "params": {}}},
            "parallel_scheme": threaded,
            "results": {"path": "out/simulation", "save": True,
                        "format": {"kind": "ParquetFormat", "params": {}}}}},
    ]}, indent=2))

    print(f"wrote SDDP baseline -> {CASE}  (risk={risk}, branchings={branchings})")
    print(f"  hydros={spec.n_res} thermals={len(spec.thermal.cap)} "
          f"stages={N_STAGES} load.mean={spec.load_cycle.mean():.0f}")
    return CASE


def _cli():
    import sys
    args = [a for a in sys.argv[1:]]
    case = next((a for a in args if a in ("2ree", "4ree")), "2ree")
    risk = next((a for a in args if a in ("expectation", "cvar")), "expectation")
    export(case=case, risk=risk)


if __name__ == "__main__":
    _cli()
