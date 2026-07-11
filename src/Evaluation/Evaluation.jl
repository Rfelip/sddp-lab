module Evaluation

# Out-of-sample / historical policy evaluation for sddp-lab.
#
# This module DEDUPES the ~5 copies of the same OOS scorer that had accreted across the
# report-machine runs (run-02 oos_simulate.jl, run-02 oos_simulate_holdout.jl, run-03
# validate_oos.jl, run-04 sf_sweeps.jl:evaluate, the scenario-rep-c001 fair_oos.jl). Every
# one trained a policy then scored it on a common SDDP.Historical bank; they differed only in
# (a) how the node path is built (finite stage index vs cyclic season vs Markov (stage,state))
# and (b) how ω is fed (physical inflow for Naive/MC vs AR standardized-residual back-out).
# Those two axes are the only real variation, captured here as process/graph dispatch.
#
# Canonical held-out bank format: a `year,stage,h1,h2,...` CSV (hydro columns 1-indexed to lab
# hydro ids), one sample path per full-horizon `year`. Per-basin builders (raw ONS/WSMPI series
# -> this layout) live with each example's data, not here.

import ..Core as LabCore     # alias: bare `Core` would resolve to Julia's built-in Base.Core
using ..StochasticProcess
using ..Scenarios
import ..System
import SDDP
import CSV
import DataFrames
import Statistics

export in_sample_cost, load_historical_bank, oos_costs, oos_summary
export load_shared_inflow_bank, export_stage_trajectories

# --- in-sample cost (InSampleMonteCarlo simulations -> total cost per series) --------------
function in_sample_cost(sims)
    return [sum(stage[LabCore.TOTAL_COST] for stage in series) for series in sims]
end

# --- load a canonical held-out bank: year,stage,h1..hN CSV -> Dict{year => stage×hydro} ----
function load_historical_bank(csv_path::String)::Dict{Int,Matrix{Float64}}
    df = CSV.read(csv_path, DataFrames.DataFrame)
    years = sort(unique(df.year))
    period = maximum(df.stage)
    hydro_cols = sort(filter(c -> startswith(c, "h"), names(df)); by = c -> parse(Int, c[2:end]))
    n_hydro = length(hydro_cols)
    out = Dict{Int,Matrix{Float64}}()
    for y in years
        sub = df[df.year .== y, :]
        DataFrames.nrow(sub) == period || continue      # skip partial-horizon years
        sub = sort(sub, :stage)
        mat = zeros(period, n_hydro)
        for (h, col) in enumerate(hydro_cols)
            mat[:, h] .= sub[!, col]
        end
        out[y] = mat
    end
    return out
end

# --- node path per (process, graph) --------------------------------------------------------
# Markov: node = (stage, nearest lattice state); cyclic: node = season; finite: node = stage.
function __oos_node_path(inflow, hist::Matrix{Float64}, cyclic::Bool, period::Int)
    n_stages = size(hist, 1)
    if inflow isa StochasticProcess.MarkovChain
        return [(t, __nearest_state(inflow.states[__season(t, period, cyclic)], hist[t, :]))
                for t in 1:n_stages]
    elseif cyclic
        return [__season(t, period, cyclic) for t in 1:n_stages]
    else
        return collect(1:n_stages)
    end
end

__season(t::Int, period::Int, cyclic::Bool) = cyclic ? ((t - 1) % period) + 1 : t

function __nearest_state(centers::Matrix{Float64}, x::Vector{Float64})::Int
    best_k, best_d = 1, Inf
    for k in 1:size(centers, 1)
        d = sum((centers[k, :] .- x) .^ 2)
        d < best_d && ((best_d, best_k) = (d, k))
    end
    return best_k
end

# --- ω path per process --------------------------------------------------------------------
# Naive / MarkovChain: ω IS the physical inflow (direct INFLOW == ω link).
# AutoRegressive: ω is the STANDARDIZED residual; back it out from consecutive physical values
#   z_t = (x_t-μ_t)/σ_t − φ·(x_{t-1}-μ_{t-1})/σ_{t-1}  (single-lag; extend per lag as needed).
struct ARSeasonParams
    phi::Float64
    mean::Float64
    std::Float64
end

function load_ar_params(ar_jsonc_path::String)::Dict{Tuple{Int,Int},ARSeasonParams}
    d = Utils.read_jsonc(ar_jsonc_path, LabCore.CompositeException())
    out = Dict{Tuple{Int,Int},ARSeasonParams}()
    for hydro in d["marginal_models"]
        hid = Int(hydro["id"])
        for m in hydro["models"]
            s = Int(m["season"])
            out[(hid, s)] = ARSeasonParams(Float64(m["coefficients"][1]),
                                           Float64.(m["scale_parameters"])...)
        end
    end
    return out
end

function __omega_path(inflow, hist::Matrix{Float64}, ar_params)
    n_stages, n_hydro = size(hist)
    if inflow isa StochasticProcess.AutoRegressive && ar_params !== nothing
        out = [zeros(n_hydro) for _ in 1:n_stages]
        for t in 1:n_stages, h in 1:n_hydro
            season = t
            prev = t == 1 ? n_stages : t - 1
            p, pp = ar_params[(h, season)], ar_params[(h, prev)]
            xprev = t == 1 ? pp.mean : hist[t - 1, h]
            out[t][h] = (hist[t, h] - p.mean) / p.std - p.phi * (xprev - pp.mean) / pp.std
        end
        return out
    end
    return [hist[t, :] for t in 1:n_stages]      # Naive / MarkovChain direct feed
end

# --- fair OOS: score a trained policy on the common held-out bank --------------------------
# Returns Dict{year => total policy cost on that historical trajectory}. Same node path + ω
# feed for every method, so cost differences are policy differences (common random numbers).
function oos_costs(model, files, bank::Dict{Int,Matrix{Float64}};
                   cyclic::Bool = false, period::Int = 12, ar_params_path = nothing)
    inflow = Scenarios.get_scenarios(files).inflow.stochastic_process
    ar_params = ar_params_path === nothing ? nothing : load_ar_params(ar_params_path)
    costs = Dict{Int,Float64}()
    for (y, hist) in sort(collect(bank); by = first)
        nodes = __oos_node_path(inflow, hist, cyclic, period)
        omegas = __omega_path(inflow, hist, ar_params)
        scenario = [(nodes[t], omegas[t]) for t in eachindex(nodes)]
        sim = SDDP.simulate(model, 1, Symbol[]; sampling_scheme = SDDP.Historical([scenario], [1.0]))
        costs[y] = sum(stage[:stage_objective] for stage in sim[1])
    end
    return costs
end

# --- summarize a cost dict/vector (mean/std/CVaR95) ----------------------------------------
function oos_summary(costs)
    v = costs isa AbstractDict ? collect(values(costs)) : collect(costs)
    n = length(v)
    n == 0 && return (mean = NaN, std = NaN, cvar95 = NaN, n = 0)
    srt = sort(v)
    q = srt[max(1, Int(round(0.95 * n)))]
    return (mean = Statistics.mean(v), std = (n > 1 ? Statistics.std(v) : 0.0),
            cvar95 = Statistics.mean(srt[srt .>= q]), n = n)
end

# --- per-stage trajectory export (SDDP-LOCATIONS.md migration step 5) ----------------------
# The old path (hydrothermal-mpc/reference-split/julia_compare.jl::_stage_row) emits one row
# per (realisation, stage) in the schema that infraestrutura_cientifica/eval/mpc-vs-sddp
# imports. oos_costs/oos_summary above only return AGGREGATE cost stats. This adds the missing
# per-stage trajectory export, routed through a lab-built policy on a shared inflow bank so the
# lab can produce _eval's CSVs directly.
#
# CSV schema (matches the frozen sddp_trajectories.csv columns, hydro/ree index 0-based):
#   se_i, gh_i, sp_i, thermal_i, deficit_i, inflow_i  (i = 0 .. n_ree-1)
#   op_cost, penalty_cost, step_wall_ms, policy, realisation, stage
#
# Field reconstruction (faithful to _stage_row on the exchange_penalty=0, single-segment,
# min-generation=0 4REE variant — see example/4ree_cyclic_5bus and MIGRATION.md):
#   se_i      = STORAGE[hydro i].out         gh_i    = HYDRO_GENERATION[hydro i]
#   sp_i      = SPILLAGE[hydro i]            inflow_i = INFLOW[hydro i]
#   thermal_i = Σ THERMAL_GENERATION over thermals whose bus_id == (ree bus i)
#   deficit_i = DEFICIT[ree bus i]
#   op_cost   = Σ THERMAL_GENERATION_COST   (== Σ thermal.cost·gen, the old op_cost)
#   penalty_cost = stage_objective − op_cost (== deficit + spillage cost; exact only when the
#                  objective carries no other terms, i.e. exchange_penalty=0 and the hydro
#                  min-generation slack is inactive — both hold for the 4ree_cyclic_5bus case).

# Load a shared inflow bank in hydrothermal-mpc's `realisation,stage,ree,inflow` layout
# (the exact format of hydrothermal-mpc/output/shared_inflows.csv) into an M-vector of
# (n_stages × n_ree) matrices. Realisation/stage/ree are 1-indexed in the file.
function load_shared_inflow_bank(csv_path::String)::Vector{Matrix{Float64}}
    df = CSV.read(csv_path, DataFrames.DataFrame)
    M = maximum(df.realisation)
    T = maximum(df.stage)
    R = maximum(df.ree)
    banks = [zeros(Float64, T, R) for _ in 1:M]
    for row in eachrow(df)
        banks[row.realisation][row.stage, row.ree] = row.inflow
    end
    return banks
end

function __ree_bus_ids(system)::Vector{Int}
    # The reservoir buses, in hydro order (hydro n lives on bus hydros[n].bus_id).
    return [Int(h.bus_id) for h in System.get_hydros_entities(system)]
end

"""
    export_stage_trajectories(model, files, banks, policy_name; period=12, cyclic=true,
                              step_wall_ms=0.0) -> DataFrames.DataFrame

Forward-simulate a trained `model` on each inflow path in `banks` (as returned by
`load_shared_inflow_bank`) via `SDDP.Historical` — common random numbers, one replay per
path — and return a DataFrame of per-(realisation, stage) rows in _eval's schema. For a
cyclic policy the node visited at stage `t` is `mod1(t, period)`; for a finite policy it is
`t`. `ω` is fed as the physical inflow vector (the Naive direct INFLOW==ω convention).
"""
function export_stage_trajectories(model, files, banks::Vector{Matrix{Float64}},
                                   policy_name::String;
                                   period::Int = 12, cyclic::Bool = true,
                                   step_wall_ms::Float64 = 0.0)
    system = System.get_system(files)
    ree_bus = __ree_bus_ids(system)                 # bus id per hydro (ree)
    n_ree = length(ree_bus)
    thermal_bus = [Int(t.bus_id) for t in System.get_thermals_entities(system)]

    # Track exactly the variables the schema needs.
    vars = [LabCore.STORED_VOLUME, LabCore.HYDRO_GENERATION, LabCore.SPILLAGE,
            LabCore.INFLOW, LabCore.THERMAL_GENERATION, LabCore.THERMAL_GENERATION_COST,
            LabCore.DEFICIT]

    # Build the historical replay: one scenario per bank path.
    scenarios = Vector{Vector{Tuple{Int,Vector{Float64}}}}()
    for path in banks
        T = size(path, 1)
        scen = Tuple{Int,Vector{Float64}}[]
        for t in 1:T
            node = cyclic ? mod1(t, period) : t
            push!(scen, (node, Float64[path[t, r] for r in 1:n_ree]))
        end
        push!(scenarios, scen)
    end

    try
        SDDP.add_all_cuts(model)
    catch
    end
    t_sim = time()
    sims = SDDP.simulate(model, length(scenarios), vars;
                         sampling_scheme = SDDP.Historical(scenarios),
                         skip_undefined_variables = true)
    sim_wall = time() - t_sim
    # If caller didn't supply a per-step wall, report the amortized average (as _stage_row does).
    total_steps = sum(length(s) for s in sims)
    swall = step_wall_ms > 0 ? step_wall_ms :
            (total_steps > 0 ? sim_wall / total_steps * 1000 : 0.0)

    rows = Vector{Dict{Symbol,Any}}()
    for (m, sim) in enumerate(sims)
        for (t, sd) in enumerate(sim)
            thermal_gen = sd[LabCore.THERMAL_GENERATION]
            op_cost = sum(sd[LabCore.THERMAL_GENERATION_COST])
            penalty_cost = sd[:stage_objective] - op_cost
            deficit = sd[LabCore.DEFICIT]
            row = Dict{Symbol,Any}(
                :policy       => policy_name,
                :realisation  => m - 1,      # 0-indexed, matches Python
                :stage        => t - 1,
                :op_cost      => Float64(op_cost),
                :penalty_cost => Float64(penalty_cost),
                :step_wall_ms => swall,
            )
            for r in 1:n_ree
                bus = ree_bus[r]
                row[Symbol("se_$(r-1)")]      = __state_out(sd[LabCore.STORED_VOLUME][r])
                row[Symbol("gh_$(r-1)")]      = sd[LabCore.HYDRO_GENERATION][r]
                row[Symbol("sp_$(r-1)")]      = sd[LabCore.SPILLAGE][r]
                row[Symbol("inflow_$(r-1)")]  = sd[LabCore.INFLOW][r]
                row[Symbol("thermal_$(r-1)")] =
                    sum((thermal_gen[j] for j in eachindex(thermal_gen) if thermal_bus[j] == bus);
                        init = 0.0)
                row[Symbol("deficit_$(r-1)")] = __deficit_at(deficit, bus)
            end
            push!(rows, row)
        end
    end
    return DataFrames.DataFrame(rows)
end

# STORAGE is an SDDP.State; SDDP.simulate records the struct — we want the outgoing value
# (matches _stage_row's stored_energy[i].out). Fall back to the raw value if it's already scalar.
__state_out(v::SDDP.State) = v.out
__state_out(v) = v

# DEFICIT may come back as a Vector (indexed 1:num_buses) or a JuMP-style axis container.
__deficit_at(deficit::AbstractVector, bus::Int) = bus <= length(deficit) ? deficit[bus] : 0.0
__deficit_at(deficit, bus::Int) = try; deficit[bus]; catch; 0.0; end

function write_stage_trajectories(path::String, df::DataFrames.DataFrame)
    CSV.write(path, df)
    return path
end

using ..Utils          # for read_jsonc (kept after the exports to avoid a load-order snag)

end
