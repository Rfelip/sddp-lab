
"""
    build_model(cfg, ena_dist)

Gera `SDDP.LinearPolicyGraph` parametrizado de acordo com configuracoes de estudo e ENAs fornecidos

# Arguments

  - `cfg::ConfigData`: configuracao do estudo como retornado por `Lab.Reader.read_config()`
  - `ena_dist::Dict{Int64,Dict{Int64,Vector{Float64}}})`: dicionario de ENAs como retornado por
    `Lab.Reader.read_ena()`
"""
function __build_model(files::Vector{InputModule}, optimizer)::SDDP.PolicyGraph
    @info "Compiling model"
    graph = __build_graph(files)
    sp_builder = __generate_subproblem_builder(files)
    
    algo = get_algorithm(files)
    cut_type_sym = get_cut_type(algo)
    cut_type = (cut_type_sym == :Multi) ? SDDP.MULTI_CUT : SDDP.SINGLE_CUT
    
    model = SDDP.PolicyGraph(
        sp_builder, graph; 
        sense = :Min, 
        lower_bound = 0.0, 
        optimizer = optimizer,
        cut_type = cut_type
    )

    return model
end

function __generate_subproblem_builder(files::Vector{InputModule})::Function
    system = get_system(files)
    scenarios = get_scenarios(files)
    num_stages = get_number_of_stages(get_algorithm(files))

    set_seed!(scenarios)

    SAA = generate_saa(scenarios, num_stages)

    # Load terminal cuts if provided (FROM FILES)
    # We will implement get_terminal_cuts in Inputs or Algorithm module
    cuts_data = get_terminal_cuts(files) 

    function fun_sp_build(m::JuMP.Model, node::Integer)
        add_system_elements!(m, system)
        add_uncertainties!(m, scenarios, node)

        # TODO - this will change once we have a proper load representation
        # as an stochastic process
        __add_load_balance!(m, files, node)

        Ω_node = vec(SAA[node])
        SDDP.parameterize(m, Ω_node) do ω
            return JuMP.fix.(m[ω_INFLOW], ω)
        end

        add_system_objective!(m, system)

        # Add Terminal Cuts if this is the last stage
        if node == num_stages && cuts_data !== nothing
            __add_terminal_cuts!(m, system, cuts_data, node)
        end

        return nothing
    end

    return fun_sp_build
end

function __add_terminal_cuts!(m::JuMP.Model, system::Any, cuts::DataFrame, node::Integer)
    # Define Future Cost Variable if not already present
    # In SDDP.jl, the future cost is often represented by a variable
    # But here we are manually adding it to the objective of the last stage.
    # We should ensure we don't double count if we were to use SDDP.BellmanTerm
    # but since this is the LAST stage, it normally has no future cost.
    # We are adding one.
    
    @variable(m, TERMINAL_FUTURE_COST >= 0)
    
    # Add to objective
    JuMP.set_objective_function(m, JuMP.objective_function(m) + TERMINAL_FUTURE_COST)

    # Filter cuts for stage 1 (terminal cuts of the study)
    stage_cuts = filter(row -> row.stage == 1, cuts)
    if isempty(stage_cuts)
        @warn "No terminal cuts found for stage 1 in the provided cuts file."
        return
    end

    cut_indexes = unique(stage_cuts.cut_index)
    
    for idx in cut_indexes
        this_cut = filter(row -> row.cut_index == idx, stage_cuts)
        
        # INTERCEPT is where state_variable_name == "INTERCEPT"
        # The value is in 'state' (based on user description: "INTERCEPT is 0 ... state: Numerical values")
        # Wait, user said: "INTERCEPT is 0, STORAGE variables range from 1 to 4" 
        # "state: Numerical values ... coefficient: Numerical coefficients"
        # Usually for INTERCEPT, the coefficient is 1 and the value is in 'state' or vice-versa.
        # In NEWAVE, the intercept is often in the 'state' column when variable is 'INTERCEPT'.
        
        intercept_row = filter(row -> row.state_variable_name == "INTERCEPT", this_cut)
        if isempty(intercept_row)
            continue
        end
        intercept = intercept_row[1, :state]
        
        cut_expr = JuMP.AffExpr(intercept)
        
        storage_rows = filter(row -> row.state_variable_name == "STORAGE", this_cut)
        for s_row in eachrow(storage_rows)
            var_id = s_row.state_variable_id
            coef = s_row.coefficient
            # state_variable_id 1 to N maps to m[STORED_VOLUME][1:N]
            # We use .out because this is a cut on the state LEAVING the last stage
            JuMP.add_to_expression!(cut_expr, coef, m[STORED_VOLUME][var_id].out)
        end
        
        @constraint(m, TERMINAL_FUTURE_COST >= cut_expr)
    end
end

# TODO - this will change
function __add_load_balance!(m::JuMP.Model, files::Vector{InputModule}, node::Integer)
    system = get_system(files)
    hydros_entities = get_hydros_entities(system)
    thermals_entities = get_thermals_entities(system)
    lines_entities = get_lines_entities(system)
    scenarios = get_scenarios(files)
    bus_ids = get_ids(get_buses(system))

    num_buses = length(bus_ids)
    num_lines = length(lines_entities)
    num_hydros = length(hydros_entities)
    num_thermals = length(thermals_entities)

    m[LOAD_BALANCE] = @constraint(
        m,
        [n = 1:num_buses],
        sum(
            m[HYDRO_GENERATION][j] for
            j in 1:num_hydros if hydros_entities[j].bus_id == bus_ids[n]
        ) +
        sum(
            m[THERMAL_GENERATION][j] for
            j in 1:num_thermals if thermals_entities[j].bus_id == bus_ids[n]
        ) +
        sum(
            m[DIRECT_EXCHANGE][j] - m[REVERSE_EXCHANGE][j] for
            j in 1:num_lines if lines_entities[j].target_bus_id == bus_ids[n]
        ) +
        sum(
            m[REVERSE_EXCHANGE][j] - m[DIRECT_EXCHANGE][j] for
            j in 1:num_lines if lines_entities[j].source_bus_id == bus_ids[n]
        ) +
        m[DEFICIT][bus_ids[n]] == get_load(bus_ids[n], node, scenarios)
    )
    return nothing
end

"""
    __build_graph(cfg)

Gera um `SDDP.Graph` parametrizado de acordo com configuracoes de estudo

# Arguments

  - `cfg::ConfigData`: configuracao do estudo como retornado por `Lab.Reader.read_config()`
"""
function __build_graph(files::Vector{InputModule})
    return generate_scenario_graph(get_algorithm(files))
end

"""
    train_model(model, cfg)

Wrapper para chamada de `SDDP.train` parametrizada de acordo com configuracoes de estudo fornecidas

# Arguments

  - `model::SDDP.PolicyGraph`: modelo construido por `Lab.Study.build_model()`
  - `cfg::ConfigData`: configuracao do estudo como retornado por `Lab.Reader.read_config()`
"""
function __train_model(
    model::SDDP.PolicyGraph,
    convergence::Convergence,
    risk::RiskMeasure,
    parallel::ParallelScheme,
)
    # Debug subproblema
    # SDDP.write_subproblem_to_file(model[1], "subproblem.lp")
    @info "Evaluating policy"
    max_iterations = convergence.max_iterations
    stopping_rule = generate_stopping_rule(get_stopping_criteria(convergence))
    risk_measure = generate_risk_measure(risk)
    parallel_scheme = generate_parallel_scheme(parallel)
    return SDDP.train(
        model;
        iteration_limit = max_iterations,
        stopping_rules = [stopping_rule],
        risk_measure = risk_measure,
        parallel_scheme = parallel_scheme,
        root_node_risk_measure = risk_measure,
    )
end

function __add_cuts_from_stage!(
    cutdata::Vector{Dict{String,Any}}, cuts::DataFrame, node::Int64
)
    stage_cuts = filter(row -> row["stage"] == node, cuts)
    cut_indexes = Int64.(unique(stage_cuts[!, "cut_index"]))
    nodedata = Dict{String,Any}(
        "risk_set_cuts" => [], "node" => string(node), "multi_cuts" => []
    )
    single_cuts = Dict{String,Any}[]
    for cut_index in cut_indexes
        cut_rows = filter(row -> row["cut_index"] == cut_index, stage_cuts)
        intercept = 0.0
        states = Dict{String,Float64}()
        coefficients = Dict{String,Float64}()
        for cut_coef in eachrow(cut_rows)
            if (cut_coef["state_variable_name"] == POLICY_CUTS_OUTPUT_INTERCEPT_NAME)
                intercept += cut_coef["state"]
            else
                key = "$(cut_coef["state_variable_name"])[$(cut_coef["state_variable_id"])]"
                state_value = cut_coef["state"]
                coefficient_value = cut_coef["coefficient"]
                push!(states, key => state_value)
                push!(coefficients, key => coefficient_value)
            end
        end
        push!(
            single_cuts,
            Dict{String,Any}(
                "state" => states, "intercept" => intercept, "coefficients" => coefficients
            ),
        )
    end
    nodedata["single_cuts"] = single_cuts
    return push!(cutdata, nodedata)
end

function __load_external_cuts!(model::SDDP.PolicyGraph, cuts::DataFrame)
    jsondata = Dict{String,Any}[]
    stages = Int64.(unique(cuts[!, "stage"]))
    for stage in stages
        __add_cuts_from_stage!(jsondata, cuts, stage)
    end
    # Adds for the last node
    __add_cuts_from_stage!(jsondata, cuts, maximum(stages) + 1)
    # Writes and reads json
    jsonpath = joinpath(tempdir(), "rawcuts2.json")
    open(jsonpath, "w") do f
        JSON.print(f, jsondata)
    end
    return SDDP.read_cuts_from_file(model, jsonpath)
end

"""
    simulate_model(model, cfg)

Realiza simulacao final parametrizada de acordo com configuracoes de estudo fornecidas

# Arguments

  - `model::SDDP.PolicyGraph`: modelo construido por `Lab.Study.build_model()`
"""
function __simulate_model(
    model::SDDP.PolicyGraph,
    files::Vector{InputModule},
    number_simulated_series::Integer,
    parallel::ParallelScheme,
)::Vector{Vector{Dict{Symbol,Any}}}
    try
        SDDP.add_all_cuts(model)
    catch
        @warn "Error while adding all cuts for simulation"
    end
    sampler = generate_sampler(get_algorithm(files))
    parallel_scheme = generate_parallel_scheme(parallel)
    @info "Running simulation"
    simulation_result = SDDP.simulate(
        model,
        number_simulated_series,
        [
            THERMAL_GENERATION,
            THERMAL_GENERATION_COST,
            INFLOW,
            TURBINED_FLOW,
            SPILLAGE,
            OUTFLOW,
            HYDRO_GENERATION,
            STORED_VOLUME,
            DEFICIT,
            NET_EXCHANGE,
            VERTEX_COVERAGE_DISTANCE,
        ];
        sampling_scheme = sampler,
        custom_recorders = Dict{Symbol,Function}(
            MARGINAL_COST => (sp::JuMP.Model) -> JuMP.dual.(sp[LOAD_BALANCE]),
            WATER_VALUE => (sp::JuMP.Model) -> JuMP.dual.(sp[HYDRO_BALANCE]),
            TOTAL_COST => (sp::JuMP.Model) -> JuMP.objective_value(sp),
        ),
        parallel_scheme = parallel_scheme,
        skip_undefined_variables = true,
    )
    return simulation_result
end
