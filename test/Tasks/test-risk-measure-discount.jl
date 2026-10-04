using SDDP, HiGHS, Test
import SDDPlab: Tasks

# On a discounted cyclic graph SDDP.jl hands a risk measure arc probabilities that sum to the arc mass b < 1. Its AV@R weights sum to 1 regardless, so the AV@R share of the cut
# is not discounted and the cuts overshoot the discounted nested value V = b * rho(Z + V). Toy: one node, a dummy state, stage cost Z in {1, 2, 3, 4, 10} equiprobable, arc probability d.
# For rho = lambda_E * E + (1 - lambda_E) * AV@R_q: the exact cut is theta = b * rho(Z) / (1 - b).
function discounted_toy_cut(measure; d = 0.9, iterations = 150)
    graph = SDDP.UnicyclicGraph(d)
    model = SDDP.PolicyGraph(graph; sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, node
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
        @constraint(sp, x.out == x.in)
        SDDP.parameterize(sp, [1.0, 2.0, 3.0, 4.0, 10.0]) do z
            return SDDP.@stageobjective(sp, z)
        end
    end
    SDDP.train(model; iteration_limit = iterations, risk_measure = measure, print_level = 0, cut_deletion_minimum = -1)
    return maximum(c.intercept for c in model[1].bellman_function.global_theta.cuts)
end

@testset "tasks-risk-measure-discounted-cyclic" begin
    d = 0.9
    # CVaR(alpha = 0.15, lambda = 0.5): lambda_E = 0.5, tail mass 0.15, AV@R_0.15(Z) = 10 (the 10 has mass 0.2), E Z = 4, rho(Z) = 7
    exact_cvar = d * (0.5 * 4.0 + 0.5 * 10.0) / (1 - d)                  # 63
    @testset "lab CVaR mapping: the cut is at most the exact discounted value" begin
        cut = discounted_toy_cut(Tasks.generate_risk_measure(Tasks.CVaR(0.15, 0.5)); d = d)
        @test cut <= exact_cvar + 1e-6
        @test cut >= exact_cvar - 1e-3                                      # and it reaches it: the fixed point of the recursion
    end
    @testset "stock SDDP.EAVaR overshoots (documents the upstream behaviour the mapping avoids)" begin
        cut = discounted_toy_cut(SDDP.EAVaR(; beta = 0.15, lambda = 0.5); d = d)
        @test cut > 1.5 * exact_cvar                                          # about 2.2x: the AV@R share is not discounted
    end
    # pure AV@R: the exact cut is d * AV@R_0.15(Z) / (1 - d) = 90; stock SDDP.AVaR has no fixed point (the cut grows with the iterations)
    exact_avar = d * 10.0 / (1 - d)
    @testset "lab AVaR mapping" begin
        cut = discounted_toy_cut(Tasks.generate_risk_measure(Tasks.AVaR(0.15)); d = d)
        @test cut <= exact_avar + 1e-6
        @test cut >= exact_avar - 1e-3
        @test discounted_toy_cut(SDDP.AVaR(0.15); d = d, iterations = 60) > 2 * exact_avar
    end
end
