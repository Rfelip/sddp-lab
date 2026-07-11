"""
SDDP baseline driver for the grid-DP cross-check.

Trains the lab on a case dir and prints the converged lower bound (the scalar
the grid-DP V_0(s0) must match within a few %). Models the manual build/train
on profile_4ree.jl: strips the unsupported `constraints` key, uses the tuned
HiGHS (presolve off, dual simplex, threads=1), trains Serial (avoids the
Threaded first-train UndefRefError race).

Usage:
    julia --project=. validate/run_sddp_baseline.jl example/2ree_cyclic [iters]
"""

using Pkg
Pkg.instantiate()

using SDDPlab, SDDP, HiGHS, Statistics

case_dir = length(ARGS) >= 1 ? ARGS[1] : "example/2ree_cyclic"
iters    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500

tuned = optimizer_with_attributes(HiGHS.Optimizer,
    "presolve" => "off", "solver" => "simplex", "simplex_strategy" => 1,
    "threads" => 1, "output_flag" => false)

base = pwd()
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

tasks_list = SDDPlab.Inputs.get_tasks(files)
policy_task = nothing
for t in tasks_list
    try; SDDPlab.Tasks.get_convergence(t); global policy_task = t; break; catch; end
end
@assert policy_task !== nothing "No Policy task found"
risk = SDDPlab.Tasks.generate_risk_measure(SDDPlab.Tasks.get_risk_measure(policy_task))
algo = SDDPlab.Algorithm.get_algorithm(files)
cut_type = SDDPlab.Algorithm.get_cut_type(algo) == :Multi ? SDDP.MULTI_CUT : SDDP.SINGLE_CUT

@info "Built model: $(length(model.nodes)) nodes. Training $iters iters (Serial, tuned HiGHS)..."
t0 = time()
SDDP.train(model; iteration_limit = iters, risk_measure = risk,
    root_node_risk_measure = risk, parallel_scheme = SDDP.Serial(),
    cut_type = cut_type, print_level = 1)
elapsed = time() - t0

bound = SDDP.calculate_bound(model)
println("\n" * "="^56)
println("SDDP BASELINE  case=$(case_dir)")
println("  trained lower bound : $(round(bound; sigdigits=8))")
println("  iterations          : $iters")
println("  train time          : $(round(elapsed; digits=1))s")
println("="^56)
cd(base)
