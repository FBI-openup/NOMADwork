module Candidates

export CandidateConfig, FlowCandidate, CandidateIndex, build_candidates, travel_time, earliest_arrival

using ..IOUtil: FlowSpec, ProblemSpec, ZoneSpec
using ..Capacity: capacity_lookup

struct CandidateConfig
    base_window::Int
    expand_step::Int
    max_window::Int
    min_capacity::Float64
end

CandidateConfig(; base_window::Int = 20, expand_step::Int = 10, max_window::Int = 200, min_capacity::Float64 = 1e-6) =
    CandidateConfig(base_window, expand_step, max_window, min_capacity)

struct FlowCandidate
    flow::FlowSpec
    slots::Dict{Tuple{Int, Int}, Vector{Int}}
end

struct CandidateIndex
    flows::Dict{Int, FlowCandidate}
end

@inline function travel_time(flow::FlowSpec, landing::Tuple{Int, Int})
    return abs(flow.access.x - landing[1]) + abs(flow.access.y - landing[2])
end

@inline function earliest_arrival(flow::FlowSpec, landing::Tuple{Int, Int})
    return flow.start_time + travel_time(flow, landing)
end

function _zone_cells(zone::ZoneSpec)
    cells = Vector{Tuple{Int, Int}}()
    for x in zone.x_min:zone.x_max
        for y in zone.y_min:zone.y_max
            push!(cells, (x, y))
        end
    end
    return cells
end

function _collect_times(flow::FlowSpec, coord::Tuple{Int, Int}, horizon::Int, capacity_table, cfg::CandidateConfig)
    earliest = earliest_arrival(flow, coord)
    if earliest >= horizon
        return Int[]
    end

    selected = Int[]
    seen = Set{Int}()
    total_capacity = 0.0
    window = cfg.base_window

    function consider_until(limit)
        total = total_capacity
        for t in earliest:min(limit, horizon - 1)
            if t in seen
                continue
            end
            push!(seen, t)
            cap = capacity_lookup(capacity_table, coord, t)
            if cap > cfg.min_capacity
                push!(selected, t)
                total += cap
            end
        end
        return total
    end

    total_capacity = consider_until(earliest + window)
    while total_capacity + cfg.min_capacity < flow.volume && window < cfg.max_window && earliest + window < horizon - 1
        window = min(cfg.max_window, window + cfg.expand_step)
        total_capacity = consider_until(earliest + window)
    end

    sort!(selected)
    return selected
end

function build_candidates(problem::ProblemSpec, capacity_table, cfg::CandidateConfig)
    flow_map = Dict{Int, FlowCandidate}()
    for flow in problem.flows
        slots = Dict{Tuple{Int, Int}, Vector{Int}}()
        for cell in _zone_cells(flow.zone)
            times = _collect_times(flow, cell, problem.grid.horizon, capacity_table, cfg)
            if !isempty(times)
                slots[cell] = times
            end
        end
        flow_map[flow.id] = FlowCandidate(flow, slots)
    end
    return CandidateIndex(flow_map)
end

end # module
