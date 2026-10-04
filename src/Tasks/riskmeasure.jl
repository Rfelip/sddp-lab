# CLASS Expectation -----------------------------------------------------------------------

function Expectation(d::Dict{String,Any}, e::CompositeException)

    # Build internal objects
    valid_internals = __build_expectation_internals_from_dicts!(d, e)

    # Keys and types validation
    valid_keys_types = valid_internals && __validate_expectation_keys_types!(d, e)

    # Content validation
    valid_content = valid_keys_types && __validate_expectation_content!(d, e)

    # Consistency validation
    valid_consistency = valid_content && __validate_expectation_consistency!(d, e)

    return valid_consistency ? Expectation() : nothing
end

# CLASS WorstCase -----------------------------------------------------------------------

function WorstCase(d::Dict{String,Any}, e::CompositeException)

    # Build internal objects
    valid_internals = __build_worstcase_internals_from_dicts!(d, e)

    # Keys and types validation
    valid_keys_types = valid_internals && __validate_worstcase_keys_types!(d, e)

    # Content validation
    valid_content = valid_keys_types && __validate_worstcase_content!(d, e)

    # Consistency validation
    valid_consistency = valid_content && __validate_worstcase_consistency!(d, e)

    return valid_consistency ? WorstCase() : nothing
end

# CLASS AVaR -----------------------------------------------------------------------

function AVaR(d::Dict{String,Any}, e::CompositeException)

    # Build internal objects
    valid_internals = __build_avar_internals_from_dicts!(d, e)

    # Keys and types validation
    valid_keys_types = valid_internals && __validate_avar_keys_types!(d, e)

    # Content validation
    valid_content = valid_keys_types && __validate_avar_content!(d, e)

    # Consistency validation
    valid_consistency = valid_content && __validate_avar_consistency!(d, e)

    return valid_consistency ? AVaR(d["alpha"]) : nothing
end

# CLASS CVaR -----------------------------------------------------------------------

function CVaR(d::Dict{String,Any}, e::CompositeException)

    # Build internal objects
    valid_internals = __build_cvar_internals_from_dicts!(d, e)

    # Keys and types validation
    valid_keys_types = valid_internals && __validate_cvar_keys_types!(d, e)

    # Content validation
    valid_content = valid_keys_types && __validate_cvar_content!(d, e)

    # Consistency validation
    valid_consistency = valid_content && __validate_cvar_consistency!(d, e)

    return valid_consistency ? CVaR(d["alpha"], d["lambda"]) : nothing
end

# SDDP METHODS --------------------------------------------------------------------------

function generate_risk_measure(r::Expectation)::SDDP.AbstractRiskMeasure
    return SDDP.Expectation()
end

function generate_risk_measure(r::WorstCase)::SDDP.AbstractRiskMeasure
    return SDDP.WorstCase()
end

# SDDP.jl hands a risk measure the arc probabilities `child.probability * noise.probability`, which sum to the
# arc mass b < 1 on a discounted graph (the discount is the probability of continuing). Its AV@R weights sum to 1
# whatever b is, so the AV@R share is not discounted: on a cyclic discounted graph the cuts overshoot the discounted
# nested value (about 2x for an expectation/AV@R mixture, unbounded for pure AV@R). `DEAVaR` is the normalised
# measure: b * [lambda_E * p + (1 - lambda_E) * AV@R_q(p)] with p the arc probabilities renormalised to sum 1, so the
# weights sum to the arc mass b and the discount sits outside the risk measure: V = b * rho(Z + V).
# Uses only SDDP.jl's documented extension point; SDDP.jl itself is unchanged. See sddp-model-cards/julia/train_ckpt.jl
# and note/drafts/sddpjl-findings.md (F1) in the CIM-SDDP project.
struct DEAVaR <: SDDP.AbstractRiskMeasure
    lambda_expectation::Float64  # weight of the Expectation in the mixture (0 = pure AV@R)
    beta::Float64                # AV@R tail mass in (0, 1]
end

function SDDP.adjust_probability(
    measure::DEAVaR,
    risk_adjusted_probability::Vector{Float64},
    original_probability::Vector{Float64},
    ::Vector,
    objective_realizations::Vector{Float64},
    is_minimization::Bool,
)
    arc_mass = sum(original_probability)
    p = original_probability ./ arc_mass
    avar = zeros(length(p))
    collected = 0.0
    for i in sortperm(objective_realizations; rev = is_minimization)
        collected >= measure.beta && break
        take = min(p[i], measure.beta - collected)
        avar[i] = take / measure.beta
        collected += take
    end
    risk_adjusted_probability .=
        arc_mass .* (measure.lambda_expectation .* p .+ (1 - measure.lambda_expectation) .* avar)
    return 0.0
end

function generate_risk_measure(r::AVaR)::SDDP.AbstractRiskMeasure
    return DEAVaR(0.0, r.alpha)
end

function generate_risk_measure(r::CVaR)::SDDP.AbstractRiskMeasure
    # the lab's CVaR(alpha, lambda) is lambda_SDDP = 1 - lambda on the EXPECTATION and beta = alpha on the tail
    return DEAVaR(1 - r.lambda, r.alpha)
end

# HELPERS -------------------------------------------------------------------------------------

function __build_risk_measure!(d::Dict{String,Any}, e::CompositeException)::Bool
    valid_key_types = __validate_risk_measure_main_key_type!(d, e)
    if !valid_key_types
        return false
    end

    return __kind_factory!(@__MODULE__, d, "risk_measure", e)
end

function __cast_risk_measure_internals_from_files!(
    d::Dict{String,Any}, e::CompositeException
)::Bool
    return true
end
