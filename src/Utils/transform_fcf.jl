using Parquet
using DataFrames
using JSON
using OrderedCollections

function transform_fcf(newwave_path, decomp_path)

    # get file newave_path/out/policy/cuts.parquet
    input_file = joinpath(newwave_path, "out", "policy", "cuts.parquet")
    output_dir = joinpath(decomp_path, "data")
    output_file = joinpath(output_dir, "terminal_cuts.jsonc")

    if !isfile(input_file)
        error("Arquivo Parquet não encontrado em: $input_file")
    end

    # get columns ...
    df = DataFrame(read_parquet(input_file))
    df.cut_index = parse.(Int, string.(df.cut_index))
    sort!(df, [:stage, :state_variable_id, :cut_index])

    nested_data = OrderedDict()
    for row in eachrow(df)
        estagio_dict = get!(nested_data, row.stage, Dict())
        usina_dict = get!(estagio_dict, row.state_variable_id, Dict())
        usina_dict[row.cut_index] = [row.coefficient, row.state]
    end

    # save at decomp_path/data/terminal_cuts.jsonc
    open(output_file, "w") do io
        JSON.print(io, nested_data, 4)
    end
end