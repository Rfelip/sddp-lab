#!/usr/bin/env python3
"""Port the WSMPI-2023 São Francisco 7-plant cascade into the sddp-lab example format.
Source data: ~/Desktop/wsmpi-2023/Dados/. See Virgil/notes/saofrancisco-cascade-model.md.
Storage/turbine/spill in hm³; inflow data in m³/s → ×K(=2.628) to hm³; generation in MWmonth.
Inflow model: per-(plant,month) LogNormal fitted from the 100 real series (zeros floored)."""
import os, json, csv
import numpy as np

WS = os.path.expanduser("~/Desktop/wsmpi-2023/Dados")
OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data"); os.makedirs(DATA, exist_ok=True)
K = 2.628  # m³/s → hm³ per 730h period (convHidro)

# --- read source ---
def read_semicolon(fn):
    return list(csv.DictReader(open(os.path.join(WS, fn), encoding="utf-8-sig"), delimiter=";"))

hidro = read_semicolon("Hidro.csv")
term  = read_semicolon("Termicas.csv")
COD2ID = {int(h["CodUsina"]): i + 1 for i, h in enumerate(hidro)}   # 155→1,…,178→7
def jus(j):  # Jusante CodUsina → lab id, 0 → "-"
    j = int(j); return "-" if j == 0 else COD2ID[j]

# --- hydros.csv (the cascade) ---
with open(os.path.join(DATA, "hydros.csv"), "w") as f:
    f.write("id ,name ,downstream_id ,bus_id ,productivity ,initial_storage ,min_storage ,max_storage ,min_generation ,max_generation ,spillage_penalty\n")
    for i, h in enumerate(hidro, 1):
        f.write(f'{i:>2} ,"{h["Nome"].strip()}" ,{str(jus(h["Jusante"])):>2} ,    1 ,'
                f'{float(h["Produtividade"]):.5f} ,{float(h["VolInicial"]):.2f} ,0.0 ,'
                f'{float(h["VolUtil"]):.2f} ,0.0 ,{float(h["Potencia"]):.4f} ,0.01\n')

# --- thermals.csv ---
with open(os.path.join(DATA, "thermals.csv"), "w") as f:
    f.write("id  ,name ,bus_id ,min_generation ,max_generation ,cost\n")
    for i, t in enumerate(term, 1):
        f.write(f'{i:>3} ,"{t["Nome"].strip()}" ,     1 ,0.0 ,{float(t["PotenciaMaxima"]):.1f} ,{float(t["CVU"]):.2f}\n')

# --- buses.csv (single bus), lines.csv (none), load.csv, stages.csv ---
open(os.path.join(DATA, "buses.csv"), "w").write('id ,name ,deficit_cost\n 1 ,"SAO_FRANCISCO" ,7643.82\n')
open(os.path.join(DATA, "lines.csv"), "w").write("id ,name ,source_bus_id ,target_bus_id ,capacity ,exchange_penalty\n")
with open(os.path.join(DATA, "load.csv"), "w") as f:
    f.write("bus_id ,stage_index ,value\n")
    for s in range(1, 13):
        f.write(f"     1 ,{s:>11} ,9000.0\n")
import datetime
with open(os.path.join(DATA, "stages.csv"), "w") as f:
    f.write("index ,start_date ,end_date\n")
    d = [datetime.date(2023, m, 1) for m in range(1, 13)] + [datetime.date(2024, 1, 1)]
    for i in range(12):
        f.write(f"{i+1:>5} ,{d[i]} ,{d[i+1]} \n")

# --- inflow_scenarios.jsonc: LogNormal per (plant, month) from the 100 real series (hm³) ---
cen = read_semicolon("CenariosAfluencias.csv")
# build inflow_hm3[codusina][month0..11] = list over 100 series
inf = {int(h["CodUsina"]): [[] for _ in range(12)] for h in hidro}
for row in cen:
    u = int(row["idUsina"])
    if u not in inf: continue
    for m in range(12):
        v = float(row[str(m + 1)]) * K   # m³/s → hm³
        inf[u][m].append(v)
FLOOR = 1.0  # hm³ floor for zeros (run-of-river / dry months)
marg = []
for i, h in enumerate(hidro, 1):
    u = int(h["CodUsina"]); dists = []
    for m in range(12):
        x = np.array([max(v, FLOOR) for v in inf[u][m]])
        lx = np.log(x)
        dists.append({"season": m + 1, "kind": "LogNormal",
                      "parameters": [round(float(lx.mean()), 6), round(float(max(lx.std(ddof=1), 1e-3)), 6)]})
    marg.append({"id": i, "distributions": dists})
ident = [[1.0 if a == b else 0.0 for b in range(7)] for a in range(7)]
inflow_json = {"marginal_models": marg,
               "copulas": [{"season": m + 1, "kind": "GaussianCopula", "parameters": ident} for m in range(12)]}
json.dump(inflow_json, open(os.path.join(DATA, "inflow_scenarios.jsonc"), "w"), indent=2)

# --- control jsoncs ---
open(os.path.join(DATA, "constraints.jsonc"), "w").write("")
json.dump({"buses": {"file": "buses.csv", "default_values": {"deficit_cost": 7643.82}},
           "lines": {"file": "lines.csv", "default_values": {"exchange_penalty": 0.0}},
           "hydros": {"file": "hydros.csv", "default_values": {"downstream_id": 0, "productivity": 1.0, "spillage_penalty": 0.01}},
           "thermals": {"file": "thermals.csv", "default_values": {}}},
          open(os.path.join(DATA, "system.jsonc"), "w"), indent=2)
# cyclic infinite horizon (discount 0.9, 12-month cycle) — matches the run-03 research setup
json.dump({"scenario_graph": {"kind": "CyclicScenarioGraph",
                              "params": {"discount_rate": 0.9, "cycle_length": 12, "cycle_stage": 1, "max_depth": 120}},
           "horizon": {"kind": "ExplicitHorizon", "params": {"file": "stages.csv"}}},
          open(os.path.join(DATA, "algorithm.jsonc"), "w"), indent=2)
json.dump({"seed": 42, "initial_season": 1, "branchings": 100,
           "inflow": {"stochastic_process": {"kind": "Naive", "params": {"file": "inflow_scenarios.jsonc"}}},
           "load": {"kind": "DeterministicLoad", "params": {"file": "load.csv"}}},
          open(os.path.join(DATA, "scenarios.jsonc"), "w"), indent=2)
json.dump({"tasks": [{"kind": "Policy", "params": {
            "convergence": {"min_iterations": 10, "max_iterations": 500,
                            "stopping_criteria": {"kind": "IterationLimit", "params": {"num_iterations": 500}}},
            "risk_measure": {"kind": "Expectation", "params": {}},
            "parallel_scheme": {"kind": "Serial", "params": {}},
            "results": {"path": "out/policy", "save": True, "format": {"kind": "ParquetFormat", "params": {}}}}}]},
          open(os.path.join(DATA, "tasks.jsonc"), "w"), indent=2)
json.dump({"inputs": {"path": "data", "files": {"tasks": "tasks.jsonc", "algorithm": "algorithm.jsonc",
            "scenarios": "scenarios.jsonc", "system": "system.jsonc", "constraints": "constraints.jsonc"}}},
          open(os.path.join(OUT, "main.jsonc"), "w"), indent=2)
print("built sddp-lab/example/saofrancisco/ (7-plant São Francisco cascade, cyclic, 100 openings)")
print("hydros.csv:"); print(open(os.path.join(DATA, "hydros.csv")).read())
