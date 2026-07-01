
function __node2season(node::Int, period::Int, initial_season::Int)
    m = (initial_season + node - 1)
    if m > period
        season = m - period * Int(div(m, period + 1e-5))
    else
        season = m
    end

    return season
end

# Markovian policy graphs (MC-SDDP) label nodes (stage, markov_state); only the stage
# component maps onto a season -- delegate to the Int method above.
function __node2season(node::Tuple{Int,Int}, period::Int, initial_season::Int)
    return __node2season(node[1], period, initial_season)
end

"""
    __node_stage(node)

Return the stage-graph index of an SDDP node. Plain policy graphs label nodes with the
stage index itself; Markovian graphs label them `(stage, markov_state)`.
"""
__node_stage(node::Int) = node
__node_stage(node::Tuple{Int,Int}) = node[1]

function __lagged_season(current_season::Int, lag::Int, period::Int)
    m = current_season - lag
    if m >= 1
        lagged_season = m
    elseif (m < 1) & (m > -period)
        lagged_season = period + m
    elseif m <= -period
        lagged_season = period + Int(rem(m, period))
    end

    return lagged_season
end