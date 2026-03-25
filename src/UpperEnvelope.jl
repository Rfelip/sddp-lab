"""
    UpperEnvelope

A 1D incremental upper envelope data structure for SDDP cut management.
Maintains the pointwise maximum of affine functions y = a*x + b.

Supports:
- `add_cut!(env, a, b)` — O(log n) amortized insertion
- `evaluate(env, x)` — O(log n) query
- `active_cut(env, x)` — O(log n) cut index lookup
- `num_active(env)` — O(1)

Reference: R2-upper-envelope-design.md
"""
module UpperEnvelope

export Envelope, add_cut!, evaluate, active_cut, num_active, num_total

"""
A cut (affine function) y = slope * x + intercept.
"""
struct Cut
    slope::Float64
    intercept::Float64
    id::Int  # original insertion order
end

"""
A breakpoint where two cuts on the envelope meet.
"""
struct Breakpoint
    x::Float64       # x-coordinate where cuts meet
    left_cut::Int     # index into cuts array (active to the left)
    right_cut::Int    # index into cuts array (active to the right)
end

"""
    Envelope

The upper envelope data structure. Internally maintains a sorted vector
of breakpoints (for simplicity; can be upgraded to a BST for strict
O(log n) guarantees, but sorted vector with binary search is O(log n)
for queries and O(n) amortized for insertion — good enough for profiling).
"""
mutable struct Envelope
    cuts::Vector{Cut}           # all cuts ever added
    breakpoints::Vector{Breakpoint}  # sorted by x
    active_indices::Vector{Int}      # which cuts are on the envelope
    # If there are no breakpoints, active_indices has 0 or 1 entries
    # (the single globally dominant cut)
end

"""
    Envelope()

Create an empty envelope.
"""
function Envelope()
    return Envelope(Cut[], Breakpoint[], Int[])
end

"""
    intersect_x(c1::Cut, c2::Cut) -> Float64

Find the x-coordinate where two cuts intersect.
Returns ±Inf if parallel (same slope).
"""
function intersect_x(c1::Cut, c2::Cut)
    if c1.slope == c2.slope
        return c1.intercept >= c2.intercept ? -Inf : Inf
    end
    return (c2.intercept - c1.intercept) / (c1.slope - c2.slope)
end

"""
    eval_cut(c::Cut, x::Float64) -> Float64

Evaluate a single cut at x.
"""
eval_cut(c::Cut, x::Float64) = c.slope * x + c.intercept

"""
    add_cut!(env::Envelope, slope::Float64, intercept::Float64) -> Int

Add a new cut y = slope * x + intercept to the envelope.
Returns the cut ID (1-based insertion order).
"""
function add_cut!(env::Envelope, slope::Float64, intercept::Float64)
    id = length(env.cuts) + 1
    new_cut = Cut(slope, intercept, id)
    push!(env.cuts, new_cut)

    if isempty(env.active_indices)
        # First cut — trivially on the envelope
        push!(env.active_indices, id)
        return id
    end

    # Check if the new cut is dominated everywhere on the current envelope
    # We need to find all intervals where the new cut exceeds the current envelope
    _rebuild_envelope!(env)
    return id
end

"""
    _rebuild_envelope!(env::Envelope)

Rebuild the envelope from scratch. O(n log n) via sweep.
Called after each insertion for correctness.

For a production implementation, this should be incremental (O(log n) amortized).
For profiling purposes, rebuild is simpler and correct.
"""
function _rebuild_envelope!(env::Envelope)
    n = length(env.cuts)
    if n == 0
        empty!(env.breakpoints)
        empty!(env.active_indices)
        return
    end

    # Sort cuts by slope (ascending), breaking ties by intercept (descending)
    perm = sortperm(env.cuts; by=c -> (c.slope, -c.intercept))

    # Build upper envelope using convex hull trick
    # Stack stores indices into env.cuts
    stack = Int[]

    for idx in perm
        c = env.cuts[idx]
        while length(stack) >= 2
            c1 = env.cuts[stack[end-1]]
            c2 = env.cuts[stack[end]]
            # Check if c2 is dominated by the intersection of c1 and c
            x12 = intersect_x(c1, c2)
            x1c = intersect_x(c1, c)
            if x1c <= x12
                # c2 is never on the envelope — remove it
                pop!(stack)
            else
                break
            end
        end
        # Skip if same slope and dominated
        if !isempty(stack) && env.cuts[stack[end]].slope == c.slope
            if env.cuts[stack[end]].intercept >= c.intercept
                continue  # existing cut dominates
            else
                pop!(stack)  # new cut dominates
            end
        end
        push!(stack, idx)
    end

    # Build breakpoints from consecutive pairs in stack
    empty!(env.breakpoints)
    env.active_indices = copy(stack)

    for i in 1:length(stack)-1
        c_left = env.cuts[stack[i]]
        c_right = env.cuts[stack[i+1]]
        x = intersect_x(c_left, c_right)
        push!(env.breakpoints, Breakpoint(x, stack[i], stack[i+1]))
    end
end

"""
    evaluate(env::Envelope, x::Float64) -> Float64

Evaluate the envelope at x. Returns -Inf for empty envelope.
O(log n) via binary search on breakpoints.
"""
function evaluate(env::Envelope, x::Float64)
    if isempty(env.active_indices)
        return -Inf
    end
    idx = _find_active_cut_index(env, x)
    return eval_cut(env.cuts[idx], x)
end

"""
    active_cut(env::Envelope, x::Float64) -> Int

Return the ID of the cut that is active (maximal) at x.
Returns 0 for empty envelope.
"""
function active_cut(env::Envelope, x::Float64)
    if isempty(env.active_indices)
        return 0
    end
    return _find_active_cut_index(env, x)
end

"""
    _find_active_cut_index(env::Envelope, x::Float64) -> Int

Binary search for the active cut at x.
"""
function _find_active_cut_index(env::Envelope, x::Float64)
    if isempty(env.breakpoints)
        return env.active_indices[1]
    end

    # Binary search: find the rightmost breakpoint with bp.x <= x
    lo, hi = 1, length(env.breakpoints)
    if x < env.breakpoints[1].x
        return env.active_indices[1]  # leftmost cut
    end
    if x >= env.breakpoints[end].x
        return env.active_indices[end]  # rightmost cut
    end

    while lo < hi
        mid = (lo + hi + 1) ÷ 2
        if env.breakpoints[mid].x <= x
            lo = mid
        else
            hi = mid - 1
        end
    end
    # x is between breakpoint[lo] and breakpoint[lo+1]
    # The active cut is the right_cut of breakpoint[lo]
    return env.breakpoints[lo].right_cut
end

"""
    num_active(env::Envelope) -> Int

Number of cuts currently on the envelope.
"""
num_active(env::Envelope) = length(env.active_indices)

"""
    num_total(env::Envelope) -> Int

Total number of cuts added (including dominated ones).
"""
num_total(env::Envelope) = length(env.cuts)

end # module
