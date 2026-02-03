using Parquet
using DataFrames
using Statistics
import Pkg
using Plots
using StatsPlots

# Path to the simulation output
parquet_path = raw"C:\Users\Adriana Catarina\Desktop\sddp-lab\example\4ree_Init0_Min0_Max0\out\simulation\operation_hydros.parquet"
output_dir = joinpath(@__DIR__, "outputs", "plots")
mkpath(output_dir)

println("Reading Parquet file: $parquet_path")

# Read the Parquet file into a DataFrame
df = read_parquet(parquet_path) |> DataFrame

# Filter for WATER_VALUE
water_value_df = filter(row -> row.variable_name == "WATER_VALUE", df)

# Get unique entities (reservoirs)
reservoirs = sort(unique(water_value_df.entity_id))

println("\n--- Generating Plots ---")

plots_array = []

for res_id in reservoirs
    println("Processing Reservoir $res_id...")
    
    # Filter data for this reservoir
    res_data = filter(row -> row.entity_id == res_id, water_value_df)
    
    if isempty(res_data)
        println("No data for Reservoir $res_id")
        push!(plots_array, plot(title="Res $res_id (No Data)", grid=false, showaxis=false))
        continue
    end

    # Create boxplot for distribution at each stage
    p = @df res_data boxplot(
        :stage, 
        :value, 
        title = "Reservoir $res_id", 
        xlabel = "Stage", 
        ylabel = "Water Value",
        #xlim = [0.5, 4.5],
        #ylim = [-100, -5],
        legend = false,
        outliers = false # Hiding outliers for cleaner visualization if desired, or true
    )
    push!(plots_array, p)
end

# Combine plots into a 2x2 layout
final_plot = plot(plots_array..., layout = (2, 2), size = (1200, 800))

# Save the plot
output_file = joinpath(output_dir, "water_values_distribution.pdf")
savefig(final_plot, output_file)

println("Plot saved to: $output_file")