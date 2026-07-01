# FORM VALIDATORS --------------------------------------------------------------------------

# MarkovChain is fit FROM the same marginal_models/copulas dict shape AutoRegressive uses
# (see autoregressive-validators.jl, which __build_stochastic_process! already dispatches
# to before we get here) -- so there is nothing extra to validate at the keys/types level.
# num_states/num_paths/fit_seed are read with defaults in the constructor, matching the
# thin/permissive validation style used by Naive above.
function __validate_markovchain_keys_types(d::Dict{String,Any}, e::CompositeException)::Bool
    return true
end

# CONTENT VALIDATORS -----------------------------------------------------------------------

function __validate_markovchain_content(d::Dict{String,Any}, e::CompositeException)::Bool
    return true
end

# CONSISTENCY VALIDATORS -------------------------------------------------------------------

function __validate_markovchain_consistency(d::Dict{String,Any}, e::CompositeException)::Bool
    return true
end
