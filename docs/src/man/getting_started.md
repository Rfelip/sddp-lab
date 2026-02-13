# Getting Started

This section goes over how to install `SDDPlab.jl` and its basic functionality.

## Installation

`SDDPlabjl` isn't registered as a `julia` package, so it must be installed from the GitHub repository. It is required to have julia installed on version `1.10` or higher.

On linux platforms

```bash
export JULIA_PKG_USE_CLI_GIT=true
julia
]
add git@github.com:rjmalves/sddp-lab.git
```

On Windows platforms, the following commands must be run on `PowerShell`

```powershell
$env:JULIA_PKG_USE_CLI_GIT=true
julia
]
add git@github.com:rjmalves/sddp-lab.git
```

## Usage Example

Although it is a package, `SDDPlab` is mainly intended to be used through the single entrypoint function `main()`

```julia
using SDDPlab
using GLPK

case_data = "path/to/case/data/directory"
optimizer = GLPK.Optimizer

SDDPlab.main(case_data, optimizer)
```

As shown in the example, `main()` takes two mandatoy arguments

1. `data_dir`: path to a directory containing a case's input data
2. `optimizer`: an optimizer, i.e. `GLPK.Optimizer`

`otimizer` was set to `GLPK` in the example, but any `julia` implementation of a linear program optimizer can be used, open or commercial. It the latter case, is left to the user to appropriately set up keys and licenses when running the function.

`data_dir` should be a followable path (either full or relative to the current working directory) to a directory containing all input files expected by `SDDPlab.jl`, in the appropriate structure. This filesystem is briefly discussed in the following section. <!---TODO: link to long form Input section-->

## Inputs

An input directory should follow a specific structure. Take for instance one of the example cases in the package, [1dsin_ar](https://github.com/rjmalves/sddp-lab/tree/main/example/1dsin_ar)

```
1dsin_ar/
├── main.jsonc
├── data/
│   ├── algorithm.jsonc
│   ├── buses.csv
│   ├── constraints.jsonc
│   ├── hydros.csv
│   ├── inflow_scenarios.jsonc
│   ├── lines.csv
│   ├── load.csv
│   ├── scenarios.jsonc
│   ├── stages.csv
│   ├── system.jsonc
│   ├── tasks.jsonc
│   ├── thermals.csv
│   └── thermals_with_inflex.csv
```

As can be noted, `SDDPlab` has a highly hierachical file structure, meant to completely modularize each aspect of the program's execution into a separate input file. There must **always** be a `main.jsonc` file which maps out the remaining input files. The contents of this file are as such

```json
{
    "inputs": {
        "path": "data",
        // Each file defines settings for a specific
        // component of the lab
        "files": {
            // Settings regarding the tasks that are being
            // performed in the current run
            // (policy optimization, simulation, etc.)
            "tasks": "tasks.jsonc",
            // Settings regarding the solution algorithm
            // that is used to solve the optimization problem
            // (decomposition strategy, discretization, etc.)
            "algorithm": "algorithm.jsonc",
            // Defines the scenarios that are used to the
            // optimization, including the scenario generation
            // model and its parameters
            "scenarios": "scenarios.jsonc",
            // Defines the system that is being optimized,
            // with all the hydro plants, thermal plants,
            // load buses and exchange lines.
            "system": "system.jsonc",
            // Defines the constraints that are used in some
            // problems, in a generic way
            "constraints": "constraints.jsonc"
        }
    }
}
```

The names of the directory `data` and all files it contains are flexible, meant to be spelled out in `main.jsonc`.

System elements, such as hydros, buses, etc. all get their own unique input file. These are tabular format, in which each line represents one element (i.e. one hydro, one bus, etc.)

### Terminal Cuts (Optional)

Terminal cuts allow coupling a short-horizon DECOMP problem with a long-horizon NEWAVE model by providing Benders cuts as a boundary condition on the last stage. To enable this, add the `terminal_cuts` key to the `files` block in `main.jsonc`:

```json
{
    "inputs": {
        "path": "data",
        "files": {
            "tasks": "tasks.jsonc",
            "algorithm": "algorithm.jsonc",
            "scenarios": "scenarios.jsonc",
            "system": "system.jsonc",
            "terminal_cuts": "cuts.parquet"
        }
    }
}
```

The cuts file (CSV or Parquet) must follow the SDDP.jl cut format with columns: `stage`, `cut_index`, `state_variable_name`, `state_variable_id`, `state`, `coefficient`. Each cut is reconstructed as:

```
θ ≥ intercept + Σ coefficient[k] * (x[k] - state[k])
```

where rows with `state_variable_name == "INTERCEPT"` provide the intercept (in the `state` column), and rows with `state_variable_name == "STORAGE"` provide the gradients and linearization points for each state variable.

If the cuts file contains multiple stages (e.g. the full output of a NEWAVE run), use `terminal_cuts_stage` to select which stage to apply:

```json
"terminal_cuts": "newave_cuts.parquet",
"terminal_cuts_stage": 2
```

When `terminal_cuts_stage` is omitted, all rows in the file are used (suitable for pre-filtered files where the stage has already been set to a single value).

## Outputs

Once `main()` finishes, the user will find a new directory named `out` in the `data_dir` originally passed to `main()`. Note that this output path can be modified in the `tasks.jsonc` file.

This new folder contains, by default:

1. an echo of the input data
2. a `PARQUET` file of the approximation cuts constructed during the policy computation
3. a `PARQUET` file all decicion variables monitored during simulations run on the computed policy