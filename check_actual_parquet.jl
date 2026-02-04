include("src/SDDPlab.jl")
using DataFrames

println("Reading actual cuts.parquet file...")
e = CompositeException()
try
    df = SDDPlab.Utils.read_parquet("example/4ree_decomp/data/cuts.parquet", e)
    if df !== nothing
        println("File read successfully.")
        println("First 5 rows:")
        println(first(df, 5))
        
        println("
Stages available: ", unique(df.stage))
        
        stage1_cuts = filter(row -> row.stage == 1, df)
        println("Number of rows for stage 1: ", nrow(stage1_cuts))
        println("Number of unique cut_index in stage 1: ", length(unique(stage1_cuts.cut_index)))
    else
        println("Failed to read file.")
        if !isempty(e)
            println("Errors: ", e)
        end
    end
catch err
    println("Error: ", err)
end
