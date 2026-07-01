"""
SDDP Profiling Script — 4REE Reservoir Problem

Instruments SDDP.train() with post_iteration_callback to log:
- Per-iteration: bound, simulation value, wall time, cuts generated
- Per-iteration per-node: total cuts, active cuts
- After training: TimerOutputs breakdown (print_level=2)

Usage:
    cd ~/Desktop/decomp-newave-analysis/data/reservatorio/4ree
    julia --project=~/Desktop/sddp-lab ~/Desktop/sddp-lab/profile_4ree.jl [max_iterations]
"""

using Pkg
Pkg.instantiate()

using SDDPlab
using SDDP
using HiGHS
using CSV
using DataFrames
using Statistics

# ── Config ──────────────────────────────────────────────────────────────────

OUT_DIR = joinpath(dirname(@__FILE__), "profiling_output")
mkpath(OUT_DIR)

MAX_ITERATIONS = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 2000

# ── Build model manually (bypassing constraints key issue) ──────────────────

@info "Building model..."

e = CompositeException()

# Read and patch the main.jsonc — remove "constraints" key which current sddp-lab doesn't support
d = SDDPlab.Utils.read_jsonc("main.jsonc", e)
@assert d !== nothing "Failed to read main.jsonc"

# Strip unsupported keys from files dict
files_d = d["inputs"]["files"]
for k in collect(keys(files_d))
    if k ∉ ("algorithm", "scenarios", "system", "tasks")
        @info "Stripping unsupported key from config: $k"
        delete!(files_d, k)
    end
end

# Build InputsData
inputs = SDDPlab.Inputs.InputsData(d["inputs"], e)
if inputs === nothing
    @error "InputsData construction failed"
    for ex in e.exceptions; @error ex; end
    error("Cannot build model")
end

# Build Entrypoint struct directly
entrypoint = SDDPlab.Inputs.Entrypoint(inputs, HiGHS.Optimizer)
files = SDDPlab.Inputs.get_files(entrypoint)

# Build the PolicyGraph
_build_model = getfield(SDDPlab.Tasks, Symbol("__build_model"))
model = _build_model(files, HiGHS.Optimizer)

# Extract risk measure from policy task
tasks_list = SDDPlab.Inputs.get_tasks(files)
local policy_task = nothing
for t in tasks_list
    try
        SDDPlab.Tasks.get_convergence(t)
        global policy_task = t
        break
    catch
    end
end
@assert policy_task !== nothing "No Policy task found"

risk = SDDPlab.Tasks.get_risk_measure(policy_task)
risk_measure = SDDPlab.Tasks.generate_risk_measure(risk)

# Get cut type from algorithm config
algo = SDDPlab.Algorithm.get_algorithm(files)
cut_type_sym = SDDPlab.Algorithm.get_cut_type(algo)
cut_type = (cut_type_sym == :Multi) ? SDDP.MULTI_CUT : SDDP.SINGLE_CUT

@info "Model built — $(length(model.nodes)) nodes, cut_type=$cut_type_sym, ready to train"

# ── Profiling callback ──────────────────────────────────────────────────────

const profile_rows = Vector{NamedTuple}()
const node_cut_rows = Vector{NamedTuple}()
const iter_counter = Ref(0)
const t_prev = Ref(time())
const nodes_list = sort(collect(keys(model.nodes)))

function profiling_callback(result)
    iter_counter[] += 1
    iter = iter_counter[]
    t_now = time()
    dt = t_now - t_prev[]
    t_prev[] = t_now

    n_cuts = sum(length(v) for (_, v) in result.cuts; init=0)

    push!(profile_rows, (
        iteration = iter,
        bound = result.bound,
        simulation_value = result.cumulative_value,
        num_cuts_generated = n_cuts,
        wall_time_s = dt,
        has_converged = result.has_converged,
    ))

    for nk in nodes_list
        node = model.nodes[nk]
        total = 0; active = 0
        try
            cuts_vec = node.bellman_function.global_theta.cuts
            total = length(cuts_vec)
            active = count(c -> c.constraint_ref !== nothing, cuts_vec)
        catch; end
        push!(node_cut_rows, (iteration=iter, node=nk, total_cuts=total, active_cuts=active))
    end

    if iter % 200 == 0 || iter == 1
        n1 = try length(model.nodes[nodes_list[1]].bellman_function.global_theta.cuts) catch; -1 end
        @info "[$iter/$MAX_ITERATIONS] bound=$(round(result.bound; sigdigits=6)) | stage1_cuts=$n1 | dt=$(round(dt*1000; digits=1))ms"
    end
    return nothing
end

# ── Train ───────────────────────────────────────────────────────────────────

@info "Training — $MAX_ITERATIONS iterations, GLPK, Serial"

t_prev[] = time()
train_start = time()

SDDP.train(
    model;
    iteration_limit = MAX_ITERATIONS,
    risk_measure = risk_measure,
    parallel_scheme = SDDP.Serial(),
    root_node_risk_measure = risk_measure,
    cut_type = cut_type,
    print_level = 2,
    post_iteration_callback = profiling_callback,
)

train_elapsed = time() - train_start

# ── Export ──────────────────────────────────────────────────────────────────

df_profile = DataFrame(profile_rows)
df_cuts = DataFrame(node_cut_rows)

conv_path = joinpath(OUT_DIR, "profile_convergence.csv")
cuts_path = joinpath(OUT_DIR, "profile_node_cuts.csv")
CSV.write(conv_path, df_profile)
CSV.write(cuts_path, df_cuts)

# ── Summary ─────────────────────────────────────────────────────────────────

times = df_profile.wall_time_s
n_iters = nrow(df_profile)

println("\n" * "="^60)
println("PROFILING SUMMARY — 4REE ($n_iters iterations)")
println("="^60)

println("\nTiming:")
println("  Total: $(round(train_elapsed; digits=1))s ($(round(train_elapsed/60; digits=1))min)")
println("  Mean/iter: $(round(mean(times)*1000; digits=1))ms")
println("  Median/iter: $(round(median(times)*1000; digits=1))ms")
println("  P95/iter: $(round(quantile(times, 0.95)*1000; digits=1))ms")
println("  P99/iter: $(round(quantile(times, 0.99)*1000; digits=1))ms")

n_bucket = min(200, n_iters)
first_bucket = mean(times[1:n_bucket])
last_bucket = mean(times[end-n_bucket+1:end])
println("  First $n_bucket avg: $(round(first_bucket*1000; digits=1))ms")
println("  Last $n_bucket avg: $(round(last_bucket*1000; digits=1))ms")
println("  Slowdown ratio: $(round(last_bucket/first_bucket; digits=2))x")

println("\nCuts (final):")
for nk in nodes_list
    try
        cuts_vec = model.nodes[nk].bellman_function.global_theta.cuts
        total = length(cuts_vec)
        active = count(c -> c.constraint_ref !== nothing, cuts_vec)
        println("  Node $nk: $total total ($active active, $(total-active) pruned)")
    catch
        println("  Node $nk: (could not read)")
    end
end

println("\nConvergence:")
println("  Final bound: $(round(df_profile.bound[end]; sigdigits=6))")

println("\nExported to:")
println("  $conv_path ($n_iters rows)")
println("  $cuts_path ($(nrow(df_cuts)) rows)")
println("="^60)
