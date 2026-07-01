#!/usr/bin/env julia
# profile_case.jl — automatic iteration profiler for ANY sddp-lab case.
#
# Generalizes the by-hand 4ree profiling (profile_4ree.jl + report-machine
# bpbp/drivers/{f1_profile,task0_profile}.jl) into one reusable WRAPPER: point it at any example
# dir, it builds the model, wraps SDDP.train with a per-iteration callback, and reads the native
# `model.timer_output` (TimerOutputs) for the forward / backward / calculate_bound wall-time
# breakdown the campaign extracts by hand. If a BPBP bunching build is active (]dev SDDP with the
# seam — see handoff-bpbp-lab-profiler.md), it also emits the bunching stats. Emits JSON + summary.
#
# Wraps the lab, does NOT edit core (canonical sddp-lab stays untouched; bunching lives in the
# ]dev scratch SDDP, so the BPBP block is guarded by isdefined and inert on the stock lab).
#
# Usage:
#   julia --project=. scripts/profile_case.jl --example example/4ree --iters 200 \
#         [--solver highs|glpk] [--out profile.json]

using SDDPlab
import SDDP, JSON, Statistics, JuMP, HiGHS, GLPK

function parse_args(argv)
    a = Dict{String,Any}("example" => nothing, "iters" => 200, "solver" => "highs", "out" => nothing)
    i = 1
    while i <= length(argv)
        x = argv[i]
        if     x == "--example"; a["example"] = argv[i+=1]
        elseif x == "--iters";   a["iters"] = parse(Int, argv[i+=1])
        elseif x == "--solver";  a["solver"] = argv[i+=1]
        elseif x == "--out";     a["out"] = argv[i+=1]
        else error("unknown arg: $x") end
        i += 1
    end
    a["example"] === nothing && error("need --example <dir>")
    return a
end

make_optimizer(n) = n == "highs" ?
    JuMP.optimizer_with_attributes(HiGHS.Optimizer, "presolve"=>"off", "solver"=>"simplex", "threads"=>1, "output_flag"=>false) :
    n == "glpk" ? GLPK.Optimizer : error("solver highs|glpk")

# Walk the native TimerOutputs tree: top-level section -> (seconds, ncalls, share).
function timer_sections(model)
    to = model.timer_output
    total_ns = to.accumulated_data.time
    total_ns == 0 && (total_ns = sum(c.accumulated_data.time for (_, c) in to.inner_timers; init=1))
    secs = Dict{String,Any}()
    for (name, child) in to.inner_timers
        t = child.accumulated_data.time
        secs[name] = (seconds = t / 1e9, ncalls = child.accumulated_data.ncalls,
                      share = t / total_ns)
    end
    return secs, total_ns / 1e9
end

function main()
    a = parse_args(ARGS)
    opt = make_optimizer(a["solver"])
    e = SDDPlab.Core.CompositeException()
    orig = pwd(); cd(a["example"])
    entry = SDDPlab.Inputs.Entrypoint("main.jsonc", opt, e); cd(orig)
    entry === nothing && error("$(a["example"]): bad entrypoint ($e)")
    files = SDDPlab.Inputs.get_files(entry)
    optimizer = SDDPlab.Inputs.get_optimizer(entry)
    tasks = SDDPlab.Tasks.get_tasks(files)
    ptask = tasks[findfirst(t -> isa(t, SDDPlab.Tasks.Policy), tasks)]
    model = SDDPlab.Tasks.__build_model(files, optimizer)

    # per-iteration wall + bound + cuts
    rows = Vector{NamedTuple}()
    tprev = Ref(time())
    cb = function (r)
        now = time(); dt = now - tprev[]; tprev[] = now
        push!(rows, (iter = length(rows) + 1, bound = r.bound,
                     sim = r.cumulative_value, wall_s = dt,
                     cuts = sum(length(v) for (_, v) in r.cuts; init=0)))
        nothing
    end

    println("=== profiling $(a["example"]) | $(a["iters"]) iters | $(a["solver"]) ===")
    SDDP.train(model; iteration_limit = a["iters"],
               stopping_rules = [SDDP.IterationLimit(a["iters"])],
               risk_measure = SDDPlab.Tasks.generate_risk_measure(SDDPlab.Tasks.get_risk_measure(ptask)),
               parallel_scheme = SDDPlab.Tasks.generate_parallel_scheme(SDDPlab.Tasks.get_parallel_scheme(ptask)),
               root_node_risk_measure = SDDPlab.Tasks.generate_risk_measure(SDDPlab.Tasks.get_risk_measure(ptask)),
               print_level = 0, post_iteration_callback = cb)

    secs, total_s = timer_sections(model)
    walls = [r.wall_s for r in rows]
    nb = min(50, length(walls))
    prof = (n_iters = length(rows),
            total_train_s = total_s,
            mean_ms = Statistics.mean(walls) * 1000,
            median_ms = Statistics.median(walls) * 1000,
            p95_ms = Statistics.quantile(walls, 0.95) * 1000,
            first_bucket_ms = Statistics.mean(walls[1:nb]) * 1000,
            last_bucket_ms = Statistics.mean(walls[end-nb+1:end]) * 1000,
            slowdown = Statistics.mean(walls[end-nb+1:end]) / Statistics.mean(walls[1:nb]),
            final_bound = rows[end].bound)

    println("\n-- timer_output sections (fwd/backward/calculate_bound breakdown) --")
    for (name, s) in sort(collect(secs); by = kv -> -kv[2].share)
        println("  $(rpad(name, 24)) $(round(s.seconds; digits=2))s  $(round(100*s.share; digits=1))%  ($(s.ncalls) calls)")
    end
    println("\n-- per-iteration wall --")
    println("  mean $(round(prof.mean_ms; digits=1))ms | median $(round(prof.median_ms; digits=1))ms | p95 $(round(prof.p95_ms; digits=1))ms")
    println("  first-$nb $(round(prof.first_bucket_ms; digits=1))ms -> last-$nb $(round(prof.last_bucket_ms; digits=1))ms  (slowdown $(round(prof.slowdown; digits=2))x)")
    println("  final bound $(round(prof.final_bound; sigdigits=6))")

    # BPBP bunching stats — only present in the ]dev BPBP build (inert on stock lab)
    bpbp = nothing
    if isdefined(SDDP, :bpbp_stats)
        try
            bpbp = SDDP.bpbp_stats()
            println("\n-- BPBP bunching stats --\n  ", bpbp)
        catch err
            println("\n-- BPBP present but stats errored: $err --")
        end
    end

    if a["out"] !== nothing
        open(a["out"], "w") do io
            JSON.print(io, Dict("example" => a["example"], "iters" => a["iters"], "solver" => a["solver"],
                                "profile" => Dict(pairs(prof)),
                                "timer_sections" => Dict(k => Dict(pairs(v)) for (k, v) in secs),
                                "bpbp" => bpbp), 2)
        end
        println("\nwrote $(a["out"])")
    end
end

main()
