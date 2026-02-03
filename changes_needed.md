# Changes Needed for Terminal Cost Function Support

This file tracks the necessary changes to `sddp-lab` to support an external Terminal Cost Function (TCF), typically derived from NEWAVE, as a boundary condition for the optimization (Policy) task.

## 1. Type Definitions (`src/Tasks/Tasks.jl`)
*   **Struct:** `Policy`
*   **Change:** Add a field `terminal_cuts::Union{String, Nothing}`.
*   **Reason:** To store the path to the CSV file containing the cuts (intercepts/coefficients) for the end of the horizon.

## 2. Validation (`src/Tasks/taskdefinition-validators.jl`)
*   **Function:** `__validate_policy_keys_types!`
*   **Change:** Add validation for the optional `terminal_cuts` key (must be a String if present).
*   **Function:** `__validate_policy_content!`
*   **Change:** Verify that if `terminal_cuts` is provided, the file exists.

## 3. Construction (`src/Tasks/taskdefinition.jl`)
*   **Function:** `Policy(d::Dict{String,Any}, e::CompositeException)`
*   **Change:** Read `terminal_cuts` from the dictionary (default to `nothing`) and pass it to the `Policy` constructor.
*   **Function:** `run_task(t::Policy, ...)`
*   **Change:** Pass `t.terminal_cuts` to `__build_model`.

## 4. Model Building (`src/Tasks/model.jl`)
*   **Function:** `__build_model`
*   **Change:** Update signature to accept `terminal_cuts::Union{String, Nothing} = nothing`. Pass this to `__generate_subproblem_builder`.
*   **Function:** `__generate_subproblem_builder`
*   **Change:**
    *   Accept `terminal_cuts`.
    *   If `terminal_cuts` is provided, read the cuts (using existing logic or a new helper).
    *   Inside the closure `fun_sp_build(m::JuMP.Model, node::Integer)`:
        *   Check if `node == num_stages`.
        *   If so, add the cuts as constraints to the model:
            *   Variable: `FUTURE_COST >= 0` (or unbounded if cuts allow negative).
            *   Constraints: `FUTURE_COST >= cut.intercept + sum(cut.coef[i] * state[i])` for each cut.
            *   Objective: Add `FUTURE_COST` to the minimization objective.

## 5. Helper Functions (`src/Tasks/model.jl`)
*   **New/Reuse:** We might need to reuse `__load_external_cuts!` logic but adapted for adding *constraints* manually to a JuMP model, rather than using `SDDP.read_cuts_from_file` (which adds them to the PolicyGraph mechanism).
*   **Reasoning:** `SDDP.add_cut` modifies the *current* approximation. If we want a *fixed* base approximation, adding them as explicit constraints `θ >= ...` is cleaner and ensures they are always active as a lower bound.

## 6. Input Data (`example/toy_decomp/data/tasks.jsonc`)
*   **Change:** Add `"terminal_cuts": "cuts.csv"` to the policy task to test the feature.
