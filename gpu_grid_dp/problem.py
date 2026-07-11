"""Problem specification for the hydrothermal grid-DP spike.

A `ProblemSpec` is a small, pure-data description of a d-reservoir cyclic
hydrothermal instance, built either from the sddp-lab CSV/jsonc case files or
from an explicit toy definition. The SAME `ProblemSpec` drives:
  * the JAX grid-DP (this package), and
  * the SDDP.jl validation baseline (via `export_lab_case`), so both solvers
    see a bit-identical instance.

Units follow the lab: energy in MWmed-month (storage, inflow, generation all
in the same energy unit since productivity = 1.0 for 4ree). Cost in $/MWmed.

Reservoirs in 4ree are independent (downstream_id empty), so the default model
has no cascade routing. `downstream` is kept for forward-compatibility.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]  # the forked sddp-lab root


# --------------------------------------------------------------------------- #
# data classes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThermalStack:
    """Merit-order thermal stack collapsed to one bus (single-energy model)."""
    cost: np.ndarray          # ($/MWmed) sorted ascending, shape [n_th]
    cap: np.ndarray           # (MWmed) capacity at each step, shape [n_th]
    deficit_cost: float       # $/MWmed for unserved load


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    n_res: int
    # per-reservoir bounds (shape [n_res])
    min_storage: np.ndarray
    max_storage: np.ndarray
    max_release: np.ndarray        # turbine cap (== max generation, productivity 1)
    productivity: np.ndarray       # MWmed per unit released
    spill_penalty: np.ndarray
    downstream: np.ndarray         # -1 = sink (no cascade); else 0-based res index
    init_storage: np.ndarray
    # demand: total system load per stage in the cycle (shape [n_stages_cycle])
    load_cycle: np.ndarray
    # inflows: lognormal(mu, sigma) of the UNDERLYING normal, per (res, season)
    inflow_mu: np.ndarray          # shape [n_res, n_seasons]
    inflow_sigma: np.ndarray       # shape [n_res, n_seasons]
    thermal: ThermalStack

    @property
    def n_seasons(self) -> int:
        return self.inflow_mu.shape[1]


# --------------------------------------------------------------------------- #
# lab CSV/jsonc parsing helpers
# --------------------------------------------------------------------------- #
def _read_lab_csv(path: Path) -> list[dict]:
    """Parse the lab's space-padded, comma-separated CSV with quoted fields."""
    rows: list[dict] = []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    header = [h.strip().strip('"') for h in lines[0].split(",")]
    for ln in lines[1:]:
        cells = [c.strip().strip('"') for c in ln.split(",")]
        rows.append(dict(zip(header, cells)))
    return rows


def _strip_jsonc(text: str) -> str:
    """Remove // line comments so the jsonc parses as json (no block comments
    are used in these lab files)."""
    return "\n".join(re.sub(r"//.*$", "", ln) for ln in text.splitlines())


def _load_jsonc(path: Path) -> dict:
    return json.loads(_strip_jsonc(path.read_text()))


def build_thermal_stack(thermals_rows: list[dict], deficit_cost: float) -> ThermalStack:
    """Collapse all thermal units into one merit-order stack (sorted by cost)."""
    cost = np.array([float(r["cost"]) for r in thermals_rows], dtype=np.float64)
    cap = np.array([float(r["max_generation"]) for r in thermals_rows], dtype=np.float64)
    order = np.argsort(cost, kind="stable")
    return ThermalStack(cost=cost[order], cap=cap[order], deficit_cost=deficit_cost)


def load_lab_case(case_dir: Path, deficit_cost: float = 1000.0) -> ProblemSpec:
    """Load a full d-reservoir lab case (e.g. example/4ree) into a ProblemSpec.

    Used to build the 4-reservoir scale instance directly from the canonical
    lab data. The cyclic load is taken as the first `n_seasons` stages of
    load.csv (the lab load is 12-month periodic).
    """
    data = case_dir / "data"
    hydros = _read_lab_csv(data / "hydros.csv")
    thermals = _read_lab_csv(data / "thermals.csv")
    load_rows = _read_lab_csv(data / "load.csv")
    inflow = _load_jsonc(data / "inflow_scenarios.jsonc")

    n_res = len(hydros)

    def col(key, default=None, cast=float):
        out = []
        for r in hydros:
            v = r.get(key, "")
            out.append(cast(v) if v not in ("", "-") else default)
        return np.array(out, dtype=np.float64)

    min_storage = col("min_storage", 0.0)
    max_storage = col("max_storage")
    max_release = col("max_generation")
    productivity = col("productivity", 1.0)
    spill_penalty = col("spillage_penalty", 0.1)
    init_storage = col("initial_storage", 0.0)

    # downstream routing: lab uses ids (1-based) or '-'; map to 0-based or -1
    id_to_idx = {r["id"].strip(): i for i, r in enumerate(hydros)}
    downstream = np.full(n_res, -1, dtype=np.int64)
    for i, r in enumerate(hydros):
        d = r.get("downstream_id", "-").strip()
        if d not in ("", "-", "0"):
            downstream[i] = id_to_idx.get(d, -1)

    # demand: sum across buses per stage -> 12-month cycle
    stages = sorted({int(r["stage_index"]) for r in load_rows})
    n_seasons = 12
    load_cycle = np.zeros(n_seasons, dtype=np.float64)
    for r in load_rows:
        s = int(r["stage_index"])
        if s <= n_seasons:
            load_cycle[s - 1] += float(r["value"])

    # inflow lognormal params per (res, season)
    mm = {m["id"]: m for m in inflow["marginal_models"]}
    inflow_mu = np.zeros((n_res, n_seasons))
    inflow_sigma = np.zeros((n_res, n_seasons))
    for i, r in enumerate(hydros):
        rid = int(r["id"])
        dists = {d["season"]: d for d in mm[rid]["distributions"]}
        for season in range(1, n_seasons + 1):
            d = dists.get(season, dists[min(dists)])
            inflow_mu[i, season - 1] = d["parameters"][0]
            inflow_sigma[i, season - 1] = d["parameters"][1]

    thermal = build_thermal_stack(thermals, deficit_cost)
    return ProblemSpec(
        name=case_dir.name, n_res=n_res,
        min_storage=min_storage, max_storage=max_storage, max_release=max_release,
        productivity=productivity, spill_penalty=spill_penalty,
        downstream=downstream, init_storage=init_storage,
        load_cycle=load_cycle, inflow_mu=inflow_mu, inflow_sigma=inflow_sigma,
        thermal=thermal,
    )


# --------------------------------------------------------------------------- #
# 2-reservoir TOY (the spike's first instance)
# --------------------------------------------------------------------------- #
def build_2ree_toy(iid: bool = True) -> ProblemSpec:
    """A 2-reservoir cyclic toy derived from 4ree reservoirs 1 & 2.

    Self-contained and explicit so the SDDP.jl baseline can be generated to
    match exactly (`export_lab_case`). Reservoirs 1 & 2 keep their 4ree caps
    and inflow laws; the thermal stack and load are scaled so that mean hydro
    energy covers ~50% of demand (thermal + occasional deficit cover the rest)
    — a non-trivial water-value problem.
    """
    full = load_lab_case(LAB_ROOT / "example" / "4ree")
    pick = [0, 1]  # reservoirs 1 & 2
    n_res = len(pick)

    min_storage = full.min_storage[pick].copy()
    max_storage = full.max_storage[pick].copy()
    max_release = full.max_release[pick].copy()
    productivity = full.productivity[pick].copy()
    spill_penalty = full.spill_penalty[pick].copy()
    downstream = np.full(n_res, -1, dtype=np.int64)   # independent
    init_storage = 0.5 * (min_storage + max_storage)  # start half-full

    inflow_mu = full.inflow_mu[pick].copy()
    inflow_sigma = full.inflow_sigma[pick].copy()
    if iid:  # collapse to season-1 law for all months (iid toy)
        inflow_mu = np.repeat(inflow_mu[:, :1], 12, axis=1)
        inflow_sigma = np.repeat(inflow_sigma[:, :1], 12, axis=1)

    # mean monthly hydro energy available (lognormal mean = exp(mu+sigma^2/2)),
    # capped by turbine -> sets the demand scale.
    mean_inflow = np.exp(inflow_mu[:, 0] + 0.5 * inflow_sigma[:, 0] ** 2)
    mean_hydro = np.minimum(mean_inflow, max_release).sum()
    # demand ~ 2x mean hydro so hydro covers ~50%; keep the 4ree 12-month shape.
    load_shape = full.load_cycle / full.load_cycle.mean()
    load_cycle = 2.0 * mean_hydro * load_shape

    # thermal: take bus-1 thermals (units 1..39 in 4ree are bus 1) — a real
    # merit order, scaled so total thermal cap ~ demand (deficit is the tail).
    thermals = _read_lab_csv(LAB_ROOT / "example" / "4ree" / "data" / "thermals.csv")
    bus1 = [r for r in thermals if r["bus_id"].strip() == "1"]
    stack = build_thermal_stack(bus1, deficit_cost=1000.0)
    # scale capacities so the stack can serve ~ the residual demand comfortably
    target_cap = 1.2 * load_cycle.max()
    scale = target_cap / stack.cap.sum()
    stack = ThermalStack(cost=stack.cost, cap=stack.cap * scale,
                         deficit_cost=stack.deficit_cost)

    return ProblemSpec(
        name="2ree_cyclic", n_res=n_res,
        min_storage=min_storage, max_storage=max_storage, max_release=max_release,
        productivity=productivity, spill_penalty=spill_penalty,
        downstream=downstream, init_storage=init_storage,
        load_cycle=load_cycle, inflow_mu=inflow_mu, inflow_sigma=inflow_sigma,
        thermal=stack,
    )


def build_problem(case: str, iid: bool) -> ProblemSpec:
    if case == "2ree":
        return build_2ree_toy(iid=iid)
    if case == "4ree":
        return load_lab_case(LAB_ROOT / "example" / "4ree")
    raise ValueError(f"unknown case {case!r}")
