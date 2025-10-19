module Capacity

export CAPACITY_PATTERN, capacity_at, build_capacity_table, capacity_lookup

using ..IOUtil: ProblemSpec, UAVSpec

const CAPACITY_PATTERN = (0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0)

@inline function capacity_at(uav::UAVSpec, t::Int)
    idx = mod(uav.phase + t, length(CAPACITY_PATTERN)) + 1
    return uav.peak_bandwidth * CAPACITY_PATTERN[idx]
end

function build_capacity_table(problem::ProblemSpec)
    table = Dict{Tuple{Int, Int}, Vector{Float64}}()
    horizon = problem.grid.horizon
    for uav in problem.uavs
        caps = Vector{Float64}(undef, horizon)
        for t in 0:(horizon - 1)
            caps[t + 1] = capacity_at(uav, t)
        end
        table[(uav.x, uav.y)] = caps
    end
    return table
end

@inline function capacity_lookup(table::Dict{Tuple{Int, Int}, Vector{Float64}}, coord::Tuple{Int, Int}, t::Int)
    caps = table[coord]
    return caps[t + 1]
end

end # module
