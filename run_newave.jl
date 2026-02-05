include("src/SDDPlab.jl")

using HiGHS

optimizer = HiGHS.Optimizer
newave_4ree_reservoir_path = "example/4ree"
newave_4ree_without_reservoir_path = "example/4ree_without_reservoir"

# Run Newave 4ree reservoir
SDDPlab.main(newave_4ree_reservoir_path, optimizer)

# Run Newave 4ree without reservoir
SDDPlab.main(newave_4ree_without_reservoir_path, optimizer)