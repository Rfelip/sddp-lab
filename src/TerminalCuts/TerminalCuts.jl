module TerminalCuts

using ..Core
using ..Utils
using DataFrames

# CLASS TerminalCutsData -----------------------------------------------------------------------

struct TerminalCutsData <: InputModule
    cuts::DataFrame
    stage::Union{Integer,Nothing}
end

function TerminalCutsData(filename::String, stage::Union{Integer,Nothing}, e::CompositeException)
    ext = lowercase(splitext(filename)[2])
    df = if ext == ".parquet"
        read_parquet(filename, e)
    else
        read_csv(filename, e)
    end
    return df !== nothing ? TerminalCutsData(df, stage) : nothing
end

# GENERAL METHODS -----------------------------------------------------------------------

"""
    get_terminal_cuts(f::Vector{InputModule})::Union{TerminalCutsData, Nothing}

Return the TerminalCutsData module if present in the files vector.
"""
function get_terminal_cuts(f::Vector{InputModule})::Union{TerminalCutsData,Nothing}
    idx = findfirst(x -> isa(x, TerminalCutsData), f)
    if idx !== nothing
        return f[idx]
    end
    return nothing
end

export TerminalCutsData, get_terminal_cuts
end