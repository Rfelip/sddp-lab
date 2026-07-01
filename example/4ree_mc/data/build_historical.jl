# build_historical.jl -- closes the reproducibility gap flagged in report-machine run-02's
# own CODE-MAP.md review ("historical_scenarios.csv has no committed builder"): a pure
# column-rename/reorder of monthly_ena.csv (SE->h1, S->h2, NE->h3, N->h4, month->stage) using
# the canonical HYDROMAP from run-02's verify_provenance.py (hydro1=SE, hydro2=S, hydro3=NE,
# hydro4=N -- matched by annual mean + seasonal phase, units MWmed both sides, no rescale).
#
# Used two ways this cycle: (1) as the held-out set for the fair-OOS scorer (Naive/AR/MC all
# judged against real ONS history, not each other's own sampling scheme), and (2) as the
# fitting data for MarkovChain's historical lattice (see markovchain.jl's
# __fit_markovchain_from_historical), instead of simulating from a (positivity-problematic)
# synthetic AR process.
using CSV
using DataFrames

ENA_PATH = "/home/rsousa/Desktop/report-machine/run-02-ar-sddp-out-of-sample/experiment-ons-data/monthly_ena.csv"
OUT_PATH = joinpath(@__DIR__, "historical_scenarios.csv")

# canonical mapping, per run-02/experiment-ons-data/verify_provenance.py:12 and experiment.md
COL_TO_HYDRO = Dict("SE" => "h1", "S" => "h2", "NE" => "h3", "N" => "h4")

df = CSV.read(ENA_PATH, DataFrame)
years = sort(unique(df.year))

rows = DataFrame(year=Int[], stage=Int[], h1=Float64[], h2=Float64[], h3=Float64[], h4=Float64[])
for y in years
    sub = df[df.year .== y, :]
    nrow(sub) == 12 || continue # partial year (2026 through June) -- needs a full 12-stage horizon
    sub = sort(sub, :month)
    for k in 1:12
        push!(rows, (
            y, k,
            sub[k, "SE"], sub[k, "S"], sub[k, "NE"], sub[k, "N"],
        ))
    end
end

CSV.write(OUT_PATH, rows)
println("wrote ", OUT_PATH, " (", length(unique(rows.year)), " full years: ", minimum(rows.year), "-", maximum(rows.year), ")")
