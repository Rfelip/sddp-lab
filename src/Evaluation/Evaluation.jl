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
import SDDP
import CSV
import DataFrames
import Statistics

export in_sample_cost, load_historical_bank, oos_costs, oos_summary

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

using ..Utils          # for read_jsonc (kept after the exports to avoid a load-order snag)

end
