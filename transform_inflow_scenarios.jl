using JSON
using Distributions

function lognormal(mu, var)
    return mean(LogNormal(mu, var))
end

function std_lognormal(mu, variance)
    return std(LogNormal(mu, variance))
end

function transform(month, file_path; out_path="transformed.json")

    data = JSON.parsefile(file_path)

    for model in data["marginal_models"]
        μ_month = 0
        μ_month_next = 0
        mu_month_next = 0
        for dist in model["distributions"]
            if month == dist["season"]
                μ  = dist["parameters"][1]
                σ2 = dist["parameters"][2]
                μ_month = lognormal(μ, σ2)
            elseif (month + 1) == dist["season"]
                μ  = dist["parameters"][1]
                σ2 = dist["parameters"][2]
                μ_month_next = lognormal(μ, σ2)
                mu_month_next = std_lognormal(μ, σ2)
            end
        end

        empty!(model["distributions"])

        push!(model["distributions"], Dict(
            "season" => 1,
            "kind" => "Normal",
            "parameters" => [μ_month, 0.0]
        ))
        push!(model["distributions"], Dict(
                    "season" => 2,
                    "kind" => "Normal",
                    "parameters" => [μ_month_next, mu_month_next]
                ))
    end

    if haskey(data, "copulas")
        data["copulas"] = [
            c for c in data["copulas"] if 1 <= c["season"] <= 2
        ]
    end

    open(out_path, "w") do f
        JSON.print(f, data, 4)
    end

    return out_path
end


transform(1,
    "example/reservatorio/4ree/data/inflow_scenarios.jsonc";
    out_path = "example/reservatorio/4ree_decomp_jan/data/inflow_scenarios.jsonc")

transform(4,
    "example/reservatorio/4ree/data/inflow_scenarios.jsonc";
    out_path = "example/reservatorio/4ree_decomp_abril/data/inflow_scenarios.jsonc")

transform(5,
    "example/reservatorio/4ree/data/inflow_scenarios.jsonc";
    out_path = "example/reservatorio/4ree_decomp_maio/data/inflow_scenarios.jsonc")

transform(8,
    "example/reservatorio/4ree/data/inflow_scenarios.jsonc";
    out_path = "example/reservatorio/4ree_decomp_agosto/data/inflow_scenarios.jsonc")

# SEM RESERVATORIO
transform(1,
    "example/sem_reservatorio/4ree/data/inflow_scenarios.jsonc";
    out_path = "example/sem_reservatorio/4ree_decomp_jan/data/inflow_scenarios.jsonc")

transform(4,
    "example/sem_reservatorio/4ree/data/inflow_scenarios.jsonc";
    out_path = "example/sem_reservatorio/4ree_decomp_abril/data/inflow_scenarios.jsonc")

transform(5,
    "example/sem_reservatorio/4ree/data/inflow_scenarios.jsonc";
    out_path = "example/sem_reservatorio/4ree_decomp_maio/data/inflow_scenarios.jsonc")

transform(8,
    "example/sem_reservatorio/4ree/data/inflow_scenarios.jsonc";
    out_path = "example/sem_reservatorio/4ree_decomp_agosto/data/inflow_scenarios.jsonc")
