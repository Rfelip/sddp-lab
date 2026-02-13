# Changes Needed for Terminal Cost Function Support

This file tracks the necessary changes to `sddp-lab` to support an external Terminal Cost Function (TCF), typically derived from NEWAVE, as a boundary condition for the optimization (Policy) task.

## Status: DONE

All items below have been implemented.

## 1. Type Definitions (`src/TerminalCuts/TerminalCuts.jl`)
*   **Struct:** `TerminalCutsData`
*   ✅ `cuts::DataFrame` — the cut data
*   ✅ `stage::Union{Integer,Nothing}` — optional stage filter

## 2. Validation (`src/Inputs/inputsdata-validators.jl`)
*   ✅ Optional `terminal_cuts` key validated as `String`
*   ✅ Optional `terminal_cuts_stage` key validated as `Integer`
*   ✅ If `terminal_cuts` is provided, file existence is verified during build

## 3. Input Loading (`src/Inputs/inputsdata-validators.jl`)
*   ✅ `TerminalCutsData` constructed with filename + stage from config
*   ✅ Pushed into `files::Vector{InputModule}` only when present
*   ✅ When `terminal_cuts` is absent, nothing is added — fully optional

## 4. Model Building (`src/Tasks/model.jl`)
*   ✅ `__generate_subproblem_builder` retrieves `TerminalCutsData` from files
*   ✅ On the last stage (`node == num_stages`), calls `__add_terminal_cuts!` if cuts exist
*   ✅ `__add_terminal_cuts!` creates `TERMINAL_FUTURE_COST >= 0` variable, adds to objective
*   ✅ Cut formula follows SDDP.jl convention: `θ ≥ intercept + Σ coeff[k] * (x[k] - state[k])`
*   ✅ Stage filtering uses `terminal_cuts_stage` if provided, otherwise uses all rows

## 5. Configuration (`main.jsonc`)
*   ✅ `"terminal_cuts": "cuts.parquet"` — path to cuts file (CSV or Parquet)
*   ✅ `"terminal_cuts_stage": 2` — optional, filters to a specific stage from the cuts file
