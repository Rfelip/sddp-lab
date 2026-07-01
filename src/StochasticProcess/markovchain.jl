# CLASS MarkovChain -------------------------------------------------------------------------
#
# MC-SDDP (Lohndorf & Shapiro, 2019): inflow uncertainty represented as a per-season finite
# lattice of K Markov states, connected across seasons by a K x K transition matrix, solved
# on an SDDP.jl Markovian policy graph. This is the discretization alternative to
# AutoRegressive's state-augmentation (TS-SDDP) -- same dependence axis, different
# representation.
#
# The lattice + transitions are FIT, not configured directly: MarkovChain(d, e) reuses the
# exact same "marginal_models"/"copulas" dict shape as AutoRegressive (see
# autoregressive-validators.jl), builds the underlying periodic AR model, simulates S sample
# paths across one seasonal cycle from it, and quantizes each season's simulated draws into K
# clusters via k-means. This keeps MC fit to the SAME data-generating process as the AR
# representation it's being compared against (Lohndorf-Shapiro's controlled-comparison
# design), rather than an independently-eyeballed lattice.

struct MarkovChain <: AbstractStochasticProcess
    n_hydro::Int
    period::Int
    K::Int

    # season => K x n_hydro matrix of lattice points (row k = state k's inflow vector)
    states::Dict{Int,Matrix{Float64}}

    # season => K x K matrix; transition[season][i, j] = P(state j at season+1 | state i at season)
    transition::Dict{Int,Matrix{Float64}}

    # season => K-vector, empirical share of simulated paths landing in each state (used as
    # the root -> season-1 distribution)
    root_probs::Dict{Int,Vector{Float64}}

    # (season, state) => pool of simulated inflow vectors assigned to that state, used to
    # draw within-node SAA branches (the "residual noise" the lattice centroid alone doesn't
    # carry)
    members::Dict{Tuple{Int,Int},Vector{Vector{Float64}}}
end

function MarkovChain(d::Dict{String,Any}, e::CompositeException)::Union{MarkovChain,Nothing}
    if haskey(d, "historical_file")
        return __fit_markovchain_from_historical(d, e)
    end
    return __fit_markovchain_from_ar(d, e)
end

# Fits the lattice + transitions to SIMULATED sample paths from an underlying periodic AR
# model (see module docstring). Kept as the original, AR-controlled-comparison fitting path.
function __fit_markovchain_from_ar(d::Dict{String,Any}, e::CompositeException)::Union{MarkovChain,Nothing}
    K = Int(get(d, "num_states", 5))
    S = Int(get(d, "num_paths", 2000))
    fit_seed = Int(get(d, "fit_seed", 1))

    ar = AutoRegressive(deepcopy(d), e)
    if ar === nothing
        return nothing
    end

    rng = Random.MersenneTwister(fit_seed)
    n_hydro, period, _ = size(ar)

    paths = __simulate_ar_paths(rng, ar, S, period) # S x period x n_hydro

    return __fit_markovchain_from_paths(rng, paths, K, period, n_hydro)
end

# Fits the lattice + transitions DIRECTLY to real historical trajectories -- positive by
# construction (real observed values), no AR/positivity problem to solve at all. Each full
# calendar year in the CSV is one sample path; k-means quantizes each season's real draws, and
# the transition matrix is the EMPIRICAL year-over-year season-to-season frequency, Laplace-
# smoothed (a handful of years gives thin per-state-pair counts -- see smoothing_alpha).
#
# CSV format: year,stage,h1,h2,h3,h4 (one row per (year,stage), hydro columns 1-indexed to
# match sddp-lab hydro ids) -- see build_historical.jl for how this is derived from raw ONS
# data via the report-machine run-02 hydro<->subsystem mapping.
function __fit_markovchain_from_historical(d::Dict{String,Any}, e::CompositeException)::Union{MarkovChain,Nothing}
    K = Int(get(d, "num_states", 5))
    fit_seed = Int(get(d, "fit_seed", 1))
    smoothing_alpha = Float64(get(d, "smoothing_alpha", 1.0))

    csv_path = d["historical_file"]
    rng = Random.MersenneTwister(fit_seed)

    paths, period, n_hydro = __load_historical_paths(csv_path)

    return __fit_markovchain_from_paths(rng, paths, K, period, n_hydro; smoothing_alpha=smoothing_alpha)
end

# Shared quantization step: S x period x n_hydro sample paths (however they were obtained --
# simulated from AR, or real historical years) -> a MarkovChain. k-means per season, empirical
# (optionally Laplace-smoothed) transitions between consecutive seasons' cluster labels.
function __fit_markovchain_from_paths(
    rng::AbstractRNG, paths::Array{Float64,3}, K::Integer, period::Integer, n_hydro::Integer;
    smoothing_alpha::Float64=0.0,
)::MarkovChain
    S = size(paths, 1)

    states = Dict{Int,Matrix{Float64}}()
    labels = Dict{Int,Vector{Int}}()
    root_probs = Dict{Int,Vector{Float64}}()
    members = Dict{Tuple{Int,Int},Vector{Vector{Float64}}}()

    for season in 1:period
        X = paths[:, season, :]
        centers, lab = __kmeans(rng, X, K)
        states[season] = centers
        labels[season] = lab
        root_probs[season] = [count(==(k), lab) / S for k in 1:K]
        for k in 1:K
            idx = findall(==(k), lab)
            members[(season, k)] = [X[i, :] for i in idx]
        end
    end

    transition = Dict{Int,Matrix{Float64}}()
    for season in 1:period
        next_season = season == period ? 1 : season + 1
        transition[season] = __empirical_transition(
            labels[season], labels[next_season], K; alpha=smoothing_alpha
        )
    end

    return MarkovChain(n_hydro, period, K, states, transition, root_probs, members)
end

# Reads a `year,stage,h1,h2,...` historical CSV (see build_historical.jl) into an S x period x
# n_hydro array, one sample path per full calendar year present in the file.
function __load_historical_paths(csv_path::String)::Tuple{Array{Float64,3},Int,Int}
    df = CSV.read(csv_path, DataFrames.DataFrame)
    years = sort(unique(df.year))
    period = maximum(df.stage)
    hydro_cols = sort(filter(c -> startswith(c, "h"), names(df)); by=c -> parse(Int, c[2:end]))
    n_hydro = length(hydro_cols)

    valid_years = [y for y in years if nrow(df[df.year .== y, :]) == period]
    S = length(valid_years)
    paths = zeros(S, period, n_hydro)
    for (p, y) in enumerate(valid_years)
        sub = sort(df[df.year .== y, :], :stage)
        for (h, col) in enumerate(hydro_cols)
            paths[p, :, h] .= sub[!, col]
        end
    end

    return paths, period, n_hydro
end

# GENERAL METHODS --------------------------------------------------------------------------

function __get_ids(s::MarkovChain)::Vector{Integer}
    return collect(1:(s.n_hydro))
end

function length(s::MarkovChain)::Integer
    return s.n_hydro
end

function size(s::MarkovChain)::Tuple{Integer,Vararg{Integer}}
    return (s.n_hydro, s.period, s.K)
end

function size(s::MarkovChain, i::Int)
    return size(s)[i]
end

# SDDP METHODS -----------------------------------------------------------------------------

function __generate_saa(
    rng::AbstractRNG, s::MarkovChain, initial_season::Integer, N::Integer, B::Integer
)::Dict{Tuple{Int,Int},Vector{Vector{Float64}}}
    out = Dict{Tuple{Int,Int},Vector{Vector{Float64}}}()

    for t in 1:N
        season = __node2season(t, s.period, Int(initial_season))
        for k in 1:(s.K)
            pool = s.members[(season, k)]
            m = length(pool)
            out[(t, k)] = if m == 0
                # degenerate cluster (can happen with a small num_paths): fall back to the
                # lattice centroid as a point mass
                [s.states[season][k, :]]
            else
                [pool[rand(rng, 1:m)] for _ in 1:B]
            end
        end
    end

    return out
end

function add_inflow_uncertainty!(m::JuMP.Model, s::MarkovChain, ::Int)::JuMP.Model
    n_hydro = s.n_hydro

    m[ω_INFLOW] = @variable(m, [1:n_hydro], base_name = String(ω_INFLOW))

    @constraint(m, inflow_model, m[INFLOW] .== m[ω_INFLOW])

    return m
end

"""
    generate_markovian_graph(s::AbstractStochasticProcess, num_stages, initial_season)

Returns `nothing` for stagewise-independent/AR processes (the caller falls back to the
regular AlgorithmData-driven graph). For a `MarkovChain`, returns the `SDDP.Graph` built from
its fitted transition matrices, cycling the per-season matrices to cover `num_stages`.
"""
function generate_markovian_graph(
    s::AbstractStochasticProcess, num_stages::Integer, initial_season::Integer
)
    return nothing
end

function generate_markovian_graph(s::MarkovChain, num_stages::Integer, initial_season::Integer)
    transition_matrices = __generate_markovian_transition_matrices(
        s, num_stages, initial_season
    )
    return SDDP.MarkovianGraph(transition_matrices)
end

function __generate_markovian_transition_matrices(
    s::MarkovChain, num_stages::Integer, initial_season::Integer
)::Vector{Matrix{Float64}}
    mats = Vector{Matrix{Float64}}(undef, num_stages)

    season_1 = __node2season(1, s.period, Int(initial_season))
    mats[1] = reshape(s.root_probs[season_1], 1, s.K)

    for t in 2:num_stages
        season_prev = __node2season(t - 1, s.period, Int(initial_season))
        mats[t] = s.transition[season_prev]
    end

    return mats
end

# HELPERS ----------------------------------------------------------------------------------

# Simulate S independent sample paths of the periodic AR model across one seasonal cycle
# (period stages), reusing the SAME per-season AR coefficients/scales/noise the JuMP
# formulation uses in add_inflow_uncertainty! (autoregressive.jl), just evaluated in plain
# arithmetic instead of as JuMP constraints. Returns an S x period x n_hydro array.
function __simulate_ar_paths(
    rng::AbstractRNG, ar::AutoRegressive, S::Integer, period::Integer
)::Array{Float64,3}
    n_hydro = length(ar)
    max_lags = [__get_lag(uar) for uar in ar.signal_model]

    out = zeros(S, period, n_hydro)

    for p in 1:S
        # rolling per-hydro history, most-recent-lag-first; bootstrapped from the config's
        # initial_values, same as the JuMP model's STCHP initial_value
        history = [copy(ar.signal_model[n].initial_values) for n in 1:n_hydro]

        for t in 1:period
            season = t
            ar_coefs = __get_ar_parameters(ar, season, true)
            lag_scales = __get_lag_scales(ar, season)
            scales = __get_ar_scale(ar, season)

            new_x = zeros(n_hydro)
            for n in 1:n_hydro
                z = 0.0
                for l in 1:max_lags[n]
                    z += ar_coefs[n][l] * (history[n][l] - lag_scales[n][l][1]) / lag_scales[n][l][2]
                end
                eps = rand(rng, ar.noise_model.models[n].distributions[season])
                new_x[n] = scales[n][1] + scales[n][2] * (z + eps)
            end

            for n in 1:n_hydro
                out[p, t, n] = new_x[n]
                history[n] = vcat([new_x[n]], history[n][1:(end - 1)])
            end
        end
    end

    return out
end

# Minimal Lloyd's-algorithm k-means with a k-means++ initialization -- no Clustering.jl
# dependency, since this is the only place in the lab that needs it. X is S x d.
function __kmeans(
    rng::AbstractRNG, X::Matrix{Float64}, K::Integer; max_iter::Integer=100
)::Tuple{Matrix{Float64},Vector{Int}}
    S, d = size(X)
    K = min(K, S)

    idx = __kmeanspp_init(rng, X, K)
    centers = X[idx, :]
    labels = zeros(Int, S)

    for _ in 1:max_iter
        changed = false
        for i in 1:S
            best_k, best_d = 1, Inf
            for k in 1:K
                dist = sum((X[i, :] .- centers[k, :]) .^ 2)
                if dist < best_d
                    best_d = dist
                    best_k = k
                end
            end
            changed = changed || labels[i] != best_k
            labels[i] = best_k
        end

        for k in 1:K
            in_k = findall(==(k), labels)
            if !isempty(in_k)
                centers[k, :] .= vec(sum(X[in_k, :]; dims=1)) ./ length(in_k)
            end
        end

        changed || break
    end

    return centers, labels
end

function __kmeanspp_init(rng::AbstractRNG, X::Matrix{Float64}, K::Integer)::Vector{Int}
    S = size(X, 1)
    idx = Vector{Int}(undef, K)
    idx[1] = rand(rng, 1:S)

    dists = [sum((X[i, :] .- X[idx[1], :]) .^ 2) for i in 1:S]
    for k in 2:K
        total = sum(dists)
        idx[k] = if total <= 0
            rand(rng, 1:S)
        else
            rand(rng, Distributions.Categorical(dists ./ total))
        end
        for i in 1:S
            dists[i] = min(dists[i], sum((X[i, :] .- X[idx[k], :]) .^ 2))
        end
    end

    return idx
end

"""
    __empirical_transition(from_labels, to_labels, K; alpha=0.0)

Empirical season-to-season transition matrix from paired cluster labels. `alpha` is a Laplace
(additive) smoothing constant: with few sample paths (e.g. 26 historical years -> ~26
observations per row, thin for a K x K estimate), `alpha > 0` pulls sparse/noisy rows toward
uniform instead of trusting a handful of counts outright.
"""
function __empirical_transition(
    from_labels::Vector{Int}, to_labels::Vector{Int}, K::Integer; alpha::Float64=0.0
)::Matrix{Float64}
    counts = zeros(K, K)
    for (i, j) in zip(from_labels, to_labels)
        counts[i, j] += 1
    end

    P = zeros(K, K)
    for i in 1:K
        row_total = sum(counts[i, :])
        P[i, :] .= if row_total > 0 || alpha > 0
            (counts[i, :] .+ alpha) ./ (row_total + alpha * K)
        else
            fill(1.0 / K, K) # unvisited state, no smoothing requested: fall back to uniform
        end
    end

    return P
end
