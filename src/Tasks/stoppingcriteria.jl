# CLASS IterationLimit -----------------------------------------------------------------------

function IterationLimit(d::Dict{String,Any}, e::CompositeException)

    # Build internal objects
    valid_internals = __build_iteration_limit_internals_from_dicts!(d, e)

    # Keys and types validation
    valid_keys_types = valid_internals && __validate_iteration_limit_keys_types!(d, e)

    # Content validation
    valid_content = valid_keys_types && __validate_iteration_limit_content!(d, e)

    # Consistency validation
    valid_consistency = valid_content && __validate_iteration_limit_consistency!(d, e)

    return if valid_consistency
        IterationLimit(d["num_iterations"])
    else
        nothing
    end
end

# CLASS TimeLimit -----------------------------------------------------------------------

function TimeLimit(d::Dict{String,Any}, e::CompositeException)

    # Build internal objects
    valid_internals = __build_time_limit_internals_from_dicts!(d, e)

    # Keys and types validation
    valid_keys_types = valid_internals && __validate_time_limit_keys_types!(d, e)

    # Content validation
    valid_content = valid_keys_types && __validate_time_limit_content!(d, e)

    # Consistency validation
    valid_consistency = valid_content && __validate_time_limit_consistency!(d, e)

    return if valid_consistency
        TimeLimit(d["time_seconds"])
    else
        nothing
    end
end

# CLASS LowerBoundStability -----------------------------------------------------------------------

function LowerBoundStability(d::Dict{String,Any}, e::CompositeException)

    # Build internal objects
    valid_internals = __build_lower_bound_stability_internals_from_dicts!(d, e)

    # Keys and types validation
    valid_keys_types = valid_internals && __validate_lower_bound_stability_keys_types!(d, e)

    # Content validation
    valid_content = valid_keys_types && __validate_lower_bound_stability_content!(d, e)

    # Consistency validation
    valid_consistency = valid_content && __validate_lower_bound_stability_consistency!(d, e)

    return if valid_consistency
        LowerBoundStability(d["threshold"], d["num_iterations"])
    else
        nothing
    end
end

# SDDP METHODS --------------------------------------------------------------------------

function generate_stopping_rule(s::IterationLimit)::SDDP.AbstractStoppingRule
    return SDDP.IterationLimit(s.num_iterations)
end

function generate_stopping_rule(s::TimeLimit)::SDDP.AbstractStoppingRule
    return SDDP.TimeLimit(s.time_seconds)
end

"""
    RelativeBoundStalling(num_previous_iterations, threshold_percent)

Stop once the bound has moved by at most `threshold_percent / 100 * |bound|` in total over
the last `num_previous_iterations` iterations, so a card's `"threshold": 0.05` reads as 0.05%.

SDDP.jl's `BoundStalling` differs twice: its tolerance is absolute (a threshold of 0.05 never
fired on a bound of order 1e8, and every training ran to its iteration limit), and it tests
each single-iteration step, which lets a slow steady climb stop training early (on `base`:
iteration 191 at 96.4% of the final bound, against 568 at 98.6% with the window).
"""
struct RelativeBoundStalling <: SDDP.AbstractStoppingRule
    num_previous_iterations::Int
    threshold_percent::Float64
end

SDDP.stopping_rule_status(::RelativeBoundStalling) = :bound_stalling

function SDDP.convergence_test(
    ::SDDP.PolicyGraph,
    log::Vector{SDDP.Log},
    rule::RelativeBoundStalling,
)
    n = rule.num_previous_iterations
    if length(log) < n + 1
        return false
    end
    # BoundStalling's guard: a bound that has not moved since the first iteration usually
    # means the cuts have not propagated back to the root yet, not convergence.
    if isapprox(log[1].bound, log[end].bound; atol = 1e-6)
        return false
    end
    tolerance = rule.threshold_percent / 100 * abs(log[end].bound)
    return abs(log[end].bound - log[end-n].bound) <= tolerance
end

function generate_stopping_rule(s::LowerBoundStability)::SDDP.AbstractStoppingRule
    return RelativeBoundStalling(s.num_iterations, s.threshold)
end

# HELPERS -------------------------------------------------------------------------------------

function __build_stopping_criteria!(d::Dict{String,Any}, e::CompositeException)::Bool
    valid_key_types = __validate_stopping_criteria_main_key_type!(d, e)
    if !valid_key_types
        return false
    end

    return __kind_factory!(@__MODULE__, d, "stopping_criteria", e)
end

function __cast_stopping_criteria_internals_from_files!(
    d::Dict{String,Any}, e::CompositeException
)::Bool
    return true
end
