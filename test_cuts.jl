include("src/SDDPlab.jl")
using DataFrames
using JuMP

println("Testing terminal cuts implementation...")

# 1. Create a dummy terminal cuts DataFrame in the Parquet format
# stage, cut_index, state_variable_name, state_variable_id, state, coefficient
cuts = DataFrame(
    stage = [11, 11, 11, 11, 11],
    cut_index = [1, 1, 1, 2, 2],
    state_variable_name = ["INTERCEPT", "STORAGE", "STORAGE", "INTERCEPT", "STORAGE"],
    state_variable_id = [0, 1, 2, 0, 1],
    state = [1000.0, 50.0, 60.0, 1200.0, 55.0],
    coefficient = [1.0, -10.0, -20.0, 1.0, -15.0]
)

# 2. Mock system - we just need it to not crash
system = nothing 

# 3. Create a JuMP model and add state variables
m = JuMP.Model()
STORAGE = Symbol("STORAGE")
m[STORAGE] = [
    (in = @variable(m, base_name="S1_in"), out = @variable(m, base_name="S1_out")),
    (in = @variable(m, base_name="S2_in"), out = @variable(m, base_name="S2_out"))
]

# 4. Call __add_terminal_cuts!
println("Calling __add_terminal_cuts! for stage 11...")
try
    # It's in the Tasks module
    SDDPlab.Tasks.__add_terminal_cuts!(m, system, cuts, 11)
    println("SUCCESS: __add_terminal_cuts! executed without errors")
    
    # 5. Verify the constraints
    println("\nModel summary:")
    # Use generic list_of_constraint_types
    for (F, S) in JuMP.list_of_constraint_types(m)
        num = JuMP.num_constraints(m, F, S)
        println("Type ($F, $S): $num")
        if num > 0
            for con in JuMP.all_constraints(m, F, S)
                println("  ", JuMP.constraint_object(con))
            end
        end
    end

    # Check if S1_out and S2_out are used
    println("\nChecking variable usage in constraints...")
    all_cons = JuMP.all_constraints(m, JuMP.AffExpr, MathOptInterface.GreaterThan{Float64})
    for con in all_cons
        obj = JuMP.constraint_object(con)
        str = string(obj.func)
        if contains(str, "S1_out") || contains(str, "S2_out")
            println("Found expected 'out' variables in constraint: ", str)
        elseif contains(str, "S1_in") || contains(str, "S2_in")
            println("ERROR: Found 'in' variables instead of 'out' variables!")
        end
    end

catch err
    println("FAILURE: Error during __add_terminal_cuts!")
    rethrow(err)
end