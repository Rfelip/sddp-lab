
# INTERNAL AR TYPES ------------------------------------------------------------------------

abstract type AbstractARparameters end

function __build_ar_parameters(d, e)
    if length(d["models"]) == 1
        SimpleARparameters(d["models"][1], e)
    else
        PeriodicARparameters(d["models"], e)
    end
end

struct SimpleARparameters <: AbstractARparameters
    phis::Vector{Float64}
    scale::Vector{Float64}
    residual_variance::Float64
    season::Int
end

function SimpleARparameters(d, e)
    valid = __validate_ar_parameters_dict!(d, e)

    return if valid
        SimpleARparameters(
            d["coefficients"], d["scale_parameters"], d["residual_variance"], d["season"]
        )
    else
        nothing
    end
end

struct PeriodicARparameters <: AbstractARparameters
    parameter_set::Vector{SimpleARparameters}
end

function PeriodicARparameters(v, e)
    parameter_set = Vector{SimpleARparameters}()
    for model_dict in v
        model = SimpleARparameters(model_dict, e)
        if !isnothing(model)
            push!(parameter_set, model)
        end
    end

    PeriodicARparameters(parameter_set)
end

# SIGNAL MODEL TYPE ------------------------------------------------------------------------

struct UnivariateAutoRegressive
    id::Int
    initial_values::Vector{Float64}
    model::AbstractARparameters
end

function UnivariateAutoRegressive(d::Dict{String, Any}, e::CompositeException)
    valid = __validate_univariateautoregressive_dict!(d, e)

    if !valid
        return nothing
    end

    arp = __build_ar_parameters(d, e)
    
    # TODO: validate that init is same size as maximum lag in models

    UnivariateAutoRegressive(d["id"], d["initial_values"], arp)
end

# MAIN AR TYPE -----------------------------------------------------------------------------

struct AutoRegressive <: AbstractStochasticProcess
    signal_model::Vector{UnivariateAutoRegressive}
    noise_model::Naive
end

function AutoRegressive(d::Dict{String, Any}, e::CompositeException)
    valid = __validate_autoregressive_dict!(d, e)

    if !valid
        return nothing
    end

    signal = Vector{UnivariateAutoRegressive}()
    for marginal_model in d["marginal_models"]
        s = UnivariateAutoRegressive(marginal_model, e)
        push!(signal, s)
    end

    noise_dict = __build_noise_naive_dict(d)
    noise = Naive(d, e)

    AutoRegressive(signal, noise)
end

function __build_noise_naive_dict(d)
    naive_dict = copy(d)

    for marg_mod in naive_dict["marginal_models"]

        delete!(marg_mod, "initial_values")

        for mod in marg_mod["models"]

            delete!(mod, "scale_parameters")
            delete!(mod, "coefficients")
            pop!(mod, "residual_variance")
            mod["kind"] = "Gaussian"
            # SAA draws a STANDARD-normal innovation z~N(0,1); the multiplicative AR converts it to a
            # variance-matched lognormal multiplier ε in parameterize_inflow! (see add_inflow_uncertainty!).
            mod["parameters"] = [0.0, 1.0]

        end

        marg_mod["distributions"] = pop!(marg_mod, "models")
    end

    return naive_dict
end

# GENERAL METHODS --------------------------------------------------------------------------

function __get_ids(s::AutoRegressive)
    return map(x -> x.id, values(s.signal_model))
end

function __get_lag(arp::SimpleARparameters)
    length(arp.phis)
end

function __get_lag(arp::PeriodicARparameters)
    max_lags = [__get_lag(i) for i in arp.parameter_set]
    maximum(max_lags)
end

function __get_lag(uar::UnivariateAutoRegressive)
    __get_lag(uar.model)
end

function __get_ar_parameters(arp::SimpleARparameters)
    arp.phis
end

function __get_ar_parameters(arp::SimpleARparameters, ::Int, ::Bool)
    __get_ar_parameters(arp)
end

function __get_ar_parameters(arp::PeriodicARparameters, season::Int, pad::Bool = false)
    seasons = map(x -> x.season, arp.parameter_set)
    index = findfirst(x -> x == season, seasons)
    if pad
        aux = __get_ar_parameters(arp.parameter_set[index])
        out = zeros(Float64, __get_lag(arp))
        for i in 1:length(aux)
            out[i] += aux[i]
        end
    else 
        out = __get_ar_parameters(arp.parameter_set[index])
    end
    return out
end

function __get_ar_parameters(uar::UnivariateAutoRegressive, season::Int, pad::Bool = false)
    __get_ar_parameters(uar.model, season, pad)
end

function __get_ar_parameters(s::AutoRegressive, season::Int, pad::Bool = false)
    [__get_ar_parameters(uar, season, pad) for uar in s.signal_model]
end

function __get_ar_scale(arp::SimpleARparameters)
    arp.scale
end

function __get_ar_scale(arp::SimpleARparameters, ::Int)
    __get_ar_scale(arp)
end

function __get_ar_scale(arp::PeriodicARparameters, season::Int)
    seasons = map(x -> x.season, arp.parameter_set)
    index = findfirst(x -> x == season, seasons)
    __get_ar_scale(arp.parameter_set[index])
end

function __get_ar_scale(uar::UnivariateAutoRegressive, season::Int)
    __get_ar_scale(uar.model, season)
end

function __get_ar_scale(s::AutoRegressive, season::Int)
    [__get_ar_scale(uar, season) for uar in s.signal_model]
end

# residual variance accessor (mirrors __get_ar_scale) — used to size the multiplicative log-noise
function __get_ar_residual(arp::SimpleARparameters)
    arp.residual_variance
end
function __get_ar_residual(arp::SimpleARparameters, ::Int)
    __get_ar_residual(arp)
end
function __get_ar_residual(arp::PeriodicARparameters, season::Int)
    seasons = map(x -> x.season, arp.parameter_set)
    index = findfirst(x -> x == season, seasons)
    __get_ar_residual(arp.parameter_set[index])
end
function __get_ar_residual(uar::UnivariateAutoRegressive, season::Int)
    __get_ar_residual(uar.model, season)
end
function __get_ar_residual(s::AutoRegressive, season::Int)
    [__get_ar_residual(uar, season) for uar in s.signal_model]
end

function length(ar::SimpleARparameters)
    return 1
end

function length(ar::PeriodicARparameters)
    return length(ar.parameter_set)
end

function length(s::AutoRegressive)
    return length(s.signal_model)
end

function size(s::AutoRegressive)
    s1 = length(s)
    period = maximum([length(uar.model) for uar in s.signal_model])
    max_lags = [__get_lag(uar) for uar in s.signal_model]

    return (s1, period, max_lags)
end

function size(s::AutoRegressive, i::Int)
    return size(s)[i]
end

# SDDP METHODS -----------------------------------------------------------------------------

function __generate_saa(
    rng::AbstractRNG,
    s::AutoRegressive,
    initial_season::Integer,
    N::Integer,
    B::Integer)

    __generate_saa(rng, s.noise_model, initial_season, N, B)
    
end

function add_inflow_uncertainty!(m::JuMP.Model, s::AutoRegressive,
    season::Int)

    n_hydro, period, max_lags = size(s)
    stchp_size = sum(max_lags)
    _use_slack = get(ENV, "AR_SLACK_OFF", "0") != "1"   # A/B toggle for verification

    scales = __get_ar_scale(s, season)
    inits = vcat([uar.initial_values for uar in s.signal_model]...)

    index_t = ones(Int,length(s))
    for i in 1:(length(s) - 1)
        index_t[i+1] = sum(max_lags[1:i]) + 1
    end
    memory_states = [n for n in 1:stchp_size if !(n in index_t)]
    
    resid = __get_ar_residual(s, season)   # standardised innovation variance per hydro for this season
    # variance-match the lognormal multiplier ε to the fitted inflow variance at the mean (CV-based),
    # so ε is a TIGHT multiplier (≈ the additive model's spread) rather than a wild one.
    logvar = [log(1 + (scales[n][2] / scales[n][1])^2 * resid[n]) for n in 1:length(s)]

    # STCHP carries the LATENT AR value and may go negative: the multiplicative AR conditional mean
    #   predicted = ε·(baserhs + Σbasecoef·lag)  is negative ~8% of openings for high-CV hydros (e.g. N).
    # Flooring STCHP itself (lower_bound 0) breaks relatively-complete recourse (infeasible deep nodes in
    # the undiscounted finite horizon) AND — if floored via a memory-state slack — lets the floor PROPAGATE
    # through the AR recursion to inflate all future inflows (an exploit that halved the cyclic bound).
    # So STCHP stays latent (free, with a finite non-binding bound to kill unbounded rays); the PHYSICAL
    # inflow is floored at the INFLOW link below (AR_INFLOW_SLACK), which feeds only the current water
    # balance — no propagation, so a deficit_cost penalty cleanly prevents any phantom-water exploit.
    m[STCHP] = @variable(m,
        [n = 1:stchp_size],
        lower_bound = _use_slack ? -1.0e8 : 0.0,
        base_name = String(STCHP),
        SDDP.State,
        initial_value = inits[n])

    lagged_scales = __get_lag_scales(s, season)
    ar_coefs = __get_ar_parameters(s, season, true)

    # MULTIPLICATIVE AR (Shapiro eq. 5.7 form, guarantees non-negativity, keeps the LP linear in the
    # state so SDDP cuts stay valid):   X_out = ε · [ μ_s + σ_s Σ_l φ_l (X_lag_l − μ_lag)/σ_lag ],
    # with ε = exp(ξ − v/2), ξ ~ N(0, v) the drawn innovation (lognormal, E[ε]=1, ε>0).
    # Built here at ε=1; parameterize_inflow!(::AutoRegressive, ξ) scales coef+rhs by ε per opening
    # via set_normalized_coefficient/rhs (the noise enters as a constraint COEFFICIENT, not a fix).
    # AR_INFLOW_SLACK: the per-hydro elastic-floor slack used at the INFLOW link below (NOT here on the
    # AR memory constraint — that version let the floor propagate through the recursion and inflate future
    # inflows, an exploit that halved the cyclic bound). Penalised at deficit_cost in add_system_objective!.
    if _use_slack
        m[AR_INFLOW_SLACK] = @variable(m, [1:n_hydro], lower_bound = 0.0, base_name = String(AR_INFLOW_SLACK))
    end
    ar_cons = Vector{JuMP.ConstraintRef}(undef, n_hydro)
    ar_lagvars = Vector{Vector{JuMP.VariableRef}}(undef, n_hydro)
    ar_basecoef = Vector{Vector{Float64}}(undef, n_hydro)
    ar_baserhs = Vector{Float64}(undef, n_hydro)
    for (n, t) in enumerate(zip(ar_coefs, lagged_scales, index_t, max_lags))
        ar_c, l_s, i, m_l = t
        s_t = scales[n]
        lagvars = [m[STCHP][i + l - 1].in for l in 1:m_l]
        basecoef = [s_t[2] * ar_c[l] / l_s[l][2] for l in 1:m_l]
        baserhs = s_t[1] - sum(s_t[2] * ar_c[l] * l_s[l][1] / l_s[l][2] for l in 1:m_l)
        con = @constraint(m,
            m[STCHP][i].out - sum(basecoef[l] * lagvars[l] for l in 1:m_l) == baserhs,
            base_name = "ar_main" * string(n))
        ar_cons[n] = con
        ar_lagvars[n] = lagvars
        ar_basecoef[n] = basecoef
        ar_baserhs[n] = baserhs
    end
    # PHYSICAL inflow floored at 0 via the elastic slack:  INFLOW = STCHP.out + slack,  INFLOW ≥ 0,
    # slack ≥ 0 penalised at deficit_cost (see add_system_objective!). When STCHP.out (latent prediction)
    # < 0 the optimum sets slack = −STCHP.out so INFLOW = 0 (floored); otherwise slack = 0 and INFLOW =
    # STCHP.out. The slack feeds ONLY this link (not the AR memory), so it cannot inflate future inflows —
    # and since one unit of inflow saves ≤ deficit_cost, the deficit_cost penalty makes manufacturing
    # water strictly unprofitable. Keeps relatively-complete recourse without the propagation exploit.
    if _use_slack
        JuMP.set_lower_bound.(m[INFLOW], 0.0)
        @constraint(m, inflow[n = 1:n_hydro],
            m[INFLOW][n] == m[STCHP][index_t[n]].out + m[AR_INFLOW_SLACK][n])
    else
        @constraint(m, inflow[n = 1:n_hydro], m[INFLOW][n] == m[STCHP][index_t[n]].out)
    end

    # memory mapping of lags
    @constraint(m, ar_memory[n in memory_states], m[STCHP][n].out == m[STCHP][n - 1].in)

    m.ext[:ar_param] = (cons = ar_cons, lagvars = ar_lagvars,
        basecoef = ar_basecoef, baserhs = ar_baserhs, logvar = logvar, n_hydro = n_hydro)
    return m
end

# Noise enters the AR subproblem as a constraint COEFFICIENT (multiplicative ε), not an additive fix.
function parameterize_inflow!(m::JuMP.Model, ::AutoRegressive, omega, node::Int)
    p = m.ext[:ar_param]
    for n in 1:p.n_hydro
        # omega[n] is a standard-normal draw; ε = exp(σ_log·z − σ_log²/2), σ_log² = logvar (mean 1, >0)
        eps = exp(omega[n] * sqrt(p.logvar[n]) - p.logvar[n] / 2)
        for (l, lv) in enumerate(p.lagvars[n])
            JuMP.set_normalized_coefficient(p.cons[n], lv, -eps * p.basecoef[n][l])
        end
        JuMP.set_normalized_rhs(p.cons[n], eps * p.baserhs[n])
    end
    return nothing
end

function __get_lag_scales(s::AutoRegressive, season::Int)
    lag_scales = []
    N, P, M_Ls = size(s)
    for n in 1:N
        aux = []
        for l in 1:M_Ls[n]
            ls = __lagged_season(season, l, P)
            push!(aux, __get_ar_scale(s.signal_model[n], ls))
        end
        push!(lag_scales, aux)
    end

    return lag_scales
end
