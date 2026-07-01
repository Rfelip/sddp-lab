#!/usr/bin/env julia
# run_and_score.jl — ONE call to train an sddp-lab case and benchmark it.
#
# The single-command "run a model + benchmark" harness that had been rebuilt from scratch every
# report-machine cycle (run_matched_budget.jl, sf_sweeps.jl, run_matrix.jl, ...). Trains one or
# more example dirs at a MATCHED iteration budget (apples-to-apples), then scores each policy
# in-sample (InSampleMonteCarlo) and, if a held-out bank is given, out-of-sample on a common
# SDDP.Historical set via SDDPlab.Evaluation. Emits a readable summary + JSON.
#
# Usage:
#   julia --project=. scripts/run_and_score.jl \
#       --example example/4ree --example example/4ree_mc \
#       --iters 300 [--solver highs|glpk] [--cyclic --period 12] \
#       [--historical example/4ree_mc/data/historical_scenarios.csv] \
#       [--ar-params <ar.jsonc>] [--out results.json]
#
# Notes: --cyclic scores over each bank path's own horizon with season nodes (saofrancisco);
# omit it for finite graphs (4ree). Defaults to HiGHS (robust on degenerate CVaR LPs; GLPK can
# hang mid-simplex — learned the hard way). Matched budget bypasses each config's own
# convergence setting so the comparison is fair.

using SDDPlab
using SDDPlab.Evaluation
import SDDP, JSON, Statistics, JuMP, HiGHS, GLPK

function parse_args(argv)
    a = Dict{String,Any}("examples" => String[], "iters" => 300, "solver" => "highs",
                          "cyclic" => false, "period" => 12, "historical" => nothing,
                          "ar_params" => nothing, "out" => nothing)
    i = 1
    while i <= length(argv)
        x = argv[i]
        if     x == "--example";    push!(a["examples"], argv[i+=1])
        elseif x == "--iters";      a["iters"] = parse(Int, argv[i+=1])
        elseif x == "--solver";     a["solver"] = argv[i+=1]
        elseif x == "--period";     a["period"] = parse(Int, argv[i+=1])
        elseif x == "--historical"; a["historical"] = argv[i+=1]
        elseif x == "--ar-params";  a["ar_params"] = argv[i+=1]
        elseif x == "--out";        a["out"] = argv[i+=1]
        elseif x == "--cyclic";     a["cyclic"] = true
        else error("unknown arg: $x")
        end
        i += 1
    end
    isempty(a["examples"]) && error("need at least one --example <dir>")
    return a
end

function make_optimizer(name)
    if name == "highs"
        return JuMP.optimizer_with_attributes(HiGHS.Optimizer,
            "presolve" => "off", "solver" => "simplex", "threads" => 1, "output_flag" => false)
    elseif name == "glpk"
        return GLPK.Optimizer
    end
    error("unknown solver: $name (use highs|glpk)")
end

function run_one(dir, iters, opt, bank, a)
    e = SDDPlab.Core.CompositeException()
    orig = pwd(); cd(dir)
    entry = SDDPlab.Inputs.Entrypoint("main.jsonc", opt, e)
    cd(orig)
    entry === nothing && error("$dir: bad entrypoint ($e)")
    files = SDDPlab.Inputs.get_files(entry)
    optimizer = SDDPlab.Inputs.get_optimizer(entry)
    tasks = SDDPlab.Tasks.get_tasks(files)
    ptask = tasks[findfirst(t -> isa(t, SDDPlab.Tasks.Policy), tasks)]
    stask = tasks[findfirst(t -> isa(t, SDDPlab.Tasks.Simulation), tasks)]

    t0 = time()
    model = SDDPlab.Tasks.__build_model(files, optimizer)
    SDDP.train(model; iteration_limit = iters,
               stopping_rules = [SDDP.IterationLimit(iters)],
               risk_measure = SDDPlab.Tasks.generate_risk_measure(SDDPlab.Tasks.get_risk_measure(ptask)),
               parallel_scheme = SDDPlab.Tasks.generate_parallel_scheme(SDDPlab.Tasks.get_parallel_scheme(ptask)),
               root_node_risk_measure = SDDPlab.Tasks.generate_risk_measure(SDDPlab.Tasks.get_risk_measure(ptask)))
    train_s = round(time() - t0; digits = 1)

    sims = SDDPlab.Tasks.__simulate_model(model, files, stask.num_simulated_series, stask.parallel_scheme)
    ins = in_sample_cost(sims)
    lb = SDDP.calculate_bound(model)

    oos = nothing
    if bank !== nothing
        costs = oos_costs(model, files, bank; cyclic = a["cyclic"], period = a["period"],
                          ar_params_path = a["ar_params"])
        oos = (summary = oos_summary(costs), per_key = costs)
    end
    return (example = dir, iters = iters, lower_bound = lb, train_s = train_s,
            in_sample = oos_summary(ins), oos = oos)
end

function main()
    a = parse_args(ARGS)
    opt = make_optimizer(a["solver"])
    bank = a["historical"] === nothing ? nothing : load_historical_bank(a["historical"])
    bank !== nothing && println("held-out bank: $(length(bank)) paths from $(a["historical"])")

    results = []
    for dir in a["examples"]
        println("\n=== $dir | iters=$(a["iters"]) | solver=$(a["solver"]) ===")
        r = run_one(dir, a["iters"], opt, bank, a)
        push!(results, r)
        println("  lower_bound = $(round(r.lower_bound; digits=1))  (train $(r.train_s)s)")
        println("  in-sample   = $(round(r.in_sample.mean; digits=1)) ± $(round(r.in_sample.std; digits=1))")
        r.oos !== nothing && println("  fair-OOS    = $(round(r.oos.summary.mean; digits=1)) ± $(round(r.oos.summary.std; digits=1))  cvar95=$(round(r.oos.summary.cvar95; digits=1))  (n=$(r.oos.summary.n))")
    end

    if a["out"] !== nothing
        payload = [Dict("example" => r.example, "iters" => r.iters, "lower_bound" => r.lower_bound,
                        "in_sample" => Dict(pairs(r.in_sample)),
                        "oos" => r.oos === nothing ? nothing :
                                 Dict("summary" => Dict(pairs(r.oos.summary)), "per_key" => r.oos.per_key))
                   for r in results]
        open(a["out"], "w") do io; JSON.print(io, payload, 2); end
        println("\nwrote $(a["out"])")
    end
end

main()
