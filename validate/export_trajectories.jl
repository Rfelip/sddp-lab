"""
Per-stage SDDP trajectory exporter (SDDP-LOCATIONS.md migration step 5).

Trains a lab policy on a case dir, forward-simulates it on a shared inflow bank
(hydrothermal-mpc's `realisation,stage,ree,inflow` CSV layout) with common random
numbers, and writes per-(realisation, stage) rows in the exact schema that
infraestrutura_cientifica/eval/mpc-vs-sddp imports (se_i, gh_i, sp_i, thermal_i,
deficit_i, inflow_i, op_cost, penalty_cost, step_wall_ms, policy, realisation, stage).

This is the missing export sddp-lab's Evaluation.jl lacked (it only did aggregate
oos_costs/oos_summary). It does NOT by itself close the numerical-equivalence gate to
the frozen paper CSVs — see MIGRATION.md (scenario-provenance step 2 + training
non-determinism). It gives the lab a policy->CSV path in the right schema.

Usage:
    julia --project=. validate/export_trajectories.jl CASE_DIR BANK_CSV OUT_CSV \\
        [iters] [policy_name] [n_realisations]

    # e.g. a quick 3-path smoke test:
    julia --project=. validate/export_trajectories.jl example/4ree_cyclic_5bus \\
        ~/Desktop/RESEARCH/experimentos/hydrothermal-mpc/output/shared_inflows.csv \\
        /tmp/sddp_traj_lab.csv 30 SDDP-IH 3
"""

using Pkg
Pkg.instantiate()

using SDDPlab, SDDP, HiGHS
import DataFrames

case_dir  = ARGS[1]
bank_csv  = ARGS[2]
out_csv   = ARGS[3]
iters     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 500
policy    = length(ARGS) >= 5 ? ARGS[5] : "SDDP-IH"
n_real    = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 0   # 0 = all paths in the bank

# Tuned HiGHS: presolve off is the lab default and trains this single-segment case fine
# (the presolve crash trap in hydrothermal-mpc was specific to its 4-segment deficit
# formulation; the native single-segment 4REE model has no degenerate ub=0 columns).
tuned = optimizer_with_attributes(HiGHS.Optimizer,
    "presolve" => "off", "solver" => "simplex", "simplex_strategy" => 1,
    "threads" => 1, "output_flag" => false)

base = pwd()
bank_csv = abspath(bank_csv)
out_csv  = abspath(out_csv)
cd(case_dir)

e = CompositeException()
d = SDDPlab.Utils.read_jsonc("main.jsonc", e)
@assert d !== nothing "Failed to read main.jsonc"
for k in collect(keys(d["inputs"]["files"]))
    k in ("algorithm", "scenarios", "system", "tasks") || delete!(d["inputs"]["files"], k)
end
inputs = SDDPlab.Inputs.InputsData(d["inputs"], e)
inputs === nothing && (for ex in e.exceptions; @error ex; end; error("InputsData failed"))
entrypoint = SDDPlab.Inputs.Entrypoint(inputs, tuned)
files = SDDPlab.Inputs.get_files(entrypoint)

model = getfield(SDDPlab.Tasks, Symbol("__build_model"))(files, tuned)

# Risk measure + cut type from the case's Policy task.
tasks_list = SDDPlab.Inputs.get_tasks(files)
policy_task = nothing
for t in tasks_list
    try; SDDPlab.Tasks.get_convergence(t); global policy_task = t; break; catch; end
end
@assert policy_task !== nothing "No Policy task found"
risk = SDDPlab.Tasks.generate_risk_measure(SDDPlab.Tasks.get_risk_measure(policy_task))
algo = SDDPlab.Algorithm.get_algorithm(files)
cut_type = SDDPlab.Algorithm.get_cut_type(algo) == :Multi ? SDDP.MULTI_CUT : SDDP.SINGLE_CUT

@info "Training policy" case=case_dir iters=iters policy=policy
t0 = time()
SDDP.train(model; iteration_limit = iters, risk_measure = risk,
    root_node_risk_measure = risk, parallel_scheme = SDDP.Serial(),
    cut_type = cut_type, print_level = 1)
@info "Trained" seconds=round(time() - t0; digits=1) bound=SDDP.calculate_bound(model)

# Load the shared inflow bank and (optionally) truncate to n_real paths.
banks = SDDPlab.Evaluation.load_shared_inflow_bank(bank_csv)
if n_real > 0 && n_real < length(banks)
    banks = banks[1:n_real]
end
@info "Loaded bank" n_paths=length(banks) n_stages=size(banks[1], 1) n_ree=size(banks[1], 2)

# Cyclic if the case uses a CyclicScenarioGraph.
graph = SDDPlab.Algorithm.get_scenario_graph(algo)
cyclic = graph isa SDDPlab.Algorithm.CyclicScenarioGraph
period = cyclic ? graph.cycle_length : 12

df = SDDPlab.Evaluation.export_stage_trajectories(model, files, banks, policy;
                                                  period = period, cyclic = cyclic)
cd(base)
SDDPlab.Evaluation.write_stage_trajectories(out_csv, df)
@info "Wrote trajectories" path=out_csv rows=DataFrames.nrow(df) cols=DataFrames.ncol(df)
println("columns: ", join(sort(names(df)), ", "))
