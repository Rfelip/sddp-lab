include("src/SDDPlab.jl")

using HiGHS

#SDDPlab.main("example/4ree", HiGHS.Optimizer)
SDDPlab.main("example/4ree_without_reservoir", HiGHS.Optimizer)