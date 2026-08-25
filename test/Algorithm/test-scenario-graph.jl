import SDDPlab: Algorithm

using Dates

REGULAR_DICT = convert(Dict{String,Any}, Dict("discount_rate" => 0.05))

CYCLIC_DICT = convert(
    Dict{String,Any},
    Dict(
        "discount_rate" => 0.05, "cycle_length" => 3, "cycle_stage" => 2, "max_depth" => 5
    ),
)

@testset "algorithm-scenario-graph" begin
    @testset "regular-scenario-graph-valid" begin
        d, e = __renew(REGULAR_DICT)
        @test typeof(Algorithm.RegularScenarioGraph(d, e)) ===
            Algorithm.RegularScenarioGraph
    end

    @testset "regular-scenario-graph-invalid-discount-rate" begin
        d, e = __renew(REGULAR_DICT)
        d = __modif_key(d, "discount_rate", -0.05)
        @test Algorithm.RegularScenarioGraph(d, e) === nothing
    end

    @testset "cyclic-scenario-graph-valid" begin
        d, e = __renew(CYCLIC_DICT)
        @test typeof(Algorithm.CyclicScenarioGraph(d, e)) === Algorithm.CyclicScenarioGraph
    end

    @testset "cyclic-scenario-graph-invalid-discount-rate" begin
        d, e = __renew(CYCLIC_DICT)
        d = __modif_key(d, "discount_rate", 1.00)
        @test Algorithm.CyclicScenarioGraph(d, e) === nothing
    end

    @testset "cyclic-scenario-graph-invalid-cycle-length" begin
        d, e = __renew(CYCLIC_DICT)
        d = __modif_key(d, "cycle_length", 0)
        @test Algorithm.CyclicScenarioGraph(d, e) === nothing
    end

    @testset "cyclic-scenario-graph-invalid-cycle-stage" begin
        d, e = __renew(CYCLIC_DICT)
        d = __modif_key(d, "cycle_stage", 0)
        @test Algorithm.CyclicScenarioGraph(d, e) === nothing
    end

    @testset "cyclic-scenario-graph-invalid-max-depth" begin
        d, e = __renew(CYCLIC_DICT)
        d = __modif_key(d, "max_depth", 3)
        @test Algorithm.CyclicScenarioGraph(d, e) === nothing
    end
end
# The edge weights themselves, which nothing above this line ever looked at.
#
# SDDP.jl reads an arc weight as the discount applied on taking that arc, so a cyclic graph's
# `discount_rate` has to be spread across the cycle: the lab wrote it onto every arc directly,
# which turned a 0.9 annual rate into 0.9 per MONTH, i.e. 0.9^12 = 0.28 per year. Every case on
# disk ships discount_rate 0.9 with cycle_length 12, so every trained policy was discounting
# roughly three times too hard. Caught 2026-08-25 comparing the lab against
# `reference-split/sddp.jl`, which spreads the rate correctly.
@testset "cyclic-scenario-graph-discount-spread-over-the-cycle" begin
    annual, len = 0.9, 12
    g = Algorithm.CyclicScenarioGraph(annual, len, 1, 120)
    graph = Algorithm.generate_scenario_graph(g, len)

    monthly = annual^(1 / len)
    # Root entry is not a time step, so it carries no discount -- same as SDDP.jl's LinearGraph.
    root = only(graph.nodes[0])
    @test root[2] ≈ 1.0

    # Every arc between real stages, including the one closing the cycle, discounts one month.
    for s in 1:len
        arc = only(graph.nodes[s])
        @test arc[2] ≈ monthly
    end

    # And one lap of the cycle costs exactly the annual rate, which is the property that matters.
    lap = prod(only(graph.nodes[s])[2] for s in 1:len)
    @test lap ≈ annual
end
