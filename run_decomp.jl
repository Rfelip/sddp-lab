include("src/SDDPlab.jl")
using HiGHS
using Parquet
using DataFrames

# Transformar inflows
include("transform_inflow_scenarios.jl")

function transform_fcf(newwave_path, decomp_path, month)

    # get file newave_path/out/policy/cuts.parquet
    input_file = joinpath(newwave_path, "out", "policy", "cuts.parquet")
    output_dir = joinpath(decomp_path, "data")
    output_file = joinpath(output_dir, "cuts.parquet")

    if !isfile(input_file)
        error("Arquivo Parquet não encontrado em:")
    end

    df = DataFrame(Parquet.read_parquet(input_file))

    df_filter = filter(:stage => x -> x == month + 1, df)
    df_filter.stage .= 1

    println(decomp_path)
    println(first(df_filter, 5))

    if !isdir(output_dir)
        mkpath(output_dir)
    end

    Parquet.write_parquet(output_file, df_filter)
end

# Transformar fcf COM RESERVATORIO
transform_fcf("example/reservatorio/4ree",
                    "example/reservatorio/4ree_decomp_jan", 1)
transform_fcf("example/reservatorio/4ree",
                    "example/reservatorio/4ree_decomp_abril", 4)
transform_fcf("example/reservatorio/4ree",
                    "example/reservatorio/4ree_decomp_maio", 5)
transform_fcf("example/reservatorio/4ree",
                    "example/reservatorio/4ree_decomp_agosto", 8)

# Transformar fcf SEM RESERVATORIO
transform_fcf("example/sem_reservatorio/4ree",
                    "example/sem_reservatorio/4ree_decomp_jan", 1)                   
transform_fcf("example/sem_reservatorio/4ree",
                    "example/sem_reservatorio/4ree_decomp_abril", 4)
transform_fcf("example/sem_reservatorio/4ree",
                    "example/sem_reservatorio/4ree_decomp_maio", 5)
transform_fcf("example/sem_reservatorio/4ree",
                    "example/sem_reservatorio/4ree_decomp_agosto", 8)


optimizer = HiGHS.Optimizer

# Rodar algoritmo COM RESERVATORIO
SDDPlab.main("example/reservatorio/4ree_decomp_jan", optimizer)
SDDPlab.main("example/reservatorio/4ree_decomp_abril", optimizer)
SDDPlab.main("example/reservatorio/4ree_decomp_maio", optimizer)
SDDPlab.main("example/reservatorio/4ree_decomp_agosto", optimizer)

# Rodar algoritmo SEM RESERVATORIO
SDDPlab.main("example/sem_reservatorio/4ree_decomp_jan", optimizer)
SDDPlab.main("example/sem_reservatorio/4ree_decomp_abril", optimizer)
SDDPlab.main("example/sem_reservatorio/4ree_decomp_maio", optimizer)
SDDPlab.main("example/sem_reservatorio/4ree_decomp_agosto", optimizer)