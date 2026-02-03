# CLASS TerminalCutsData -----------------------------------------------------------------------

struct TerminalCutsData <: InputModule
    cuts::DataFrame
end

function TerminalCutsData(filename::String, e::CompositeException)
    df = read_dataframe(filename, e)
    return df !== nothing ? TerminalCutsData(df) : nothing
end

# GENERAL METHODS -----------------------------------------------------------------------

"""
get_terminal_cuts(f::Vector{InputModule})::Union{DataFrame, Nothing}

Return the terminal cuts dataframe if present.
"""
function get_terminal_cuts(f::Vector{InputModule})::Union{DataFrame,Nothing}
    # We look for the TerminalCutsData in the vector
    # This relies on get_input_module returning the struct, or nothing if not found
    # But get_input_module usually assumes 1 instance.
    # Let's check how get_input_module is implemented in Utils.
    # Assuming it returns the module or throws/returns default.
    
    # We can implement a safe search here
    idx = findfirst(x -> isa(x, TerminalCutsData), f)
    if idx !== nothing
        return f[idx].cuts
    end
    return nothing
end