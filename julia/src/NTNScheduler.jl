module NTNScheduler

export run_cli, solve_problem, default_config

include("IO.jl")
include("Capacity.jl")
include("Candidates.jl")
include("Clustering.jl")
include("Heuristics.jl")
include("Model.jl")
include("PostProcess.jl")
include("Main.jl")

using .IOUtil
using .Capacity
using .Candidates
using .Clustering
using .Heuristics
using .Model
using .PostProcess
using .Main

const default_config = Main.default_config
const solve_problem = Main.solve_problem
const run_cli = Main.run_cli

end # module
