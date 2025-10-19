module Heuristics

export WarmStart, build_warm_start

using ..IOUtil: FlowSpec, ProblemSpec, ScheduleSegment
using ..Candidates: CandidateIndex, FlowCandidate, travel_time
using ..Capacity: capacity_lookup

const EPS = 1e-6

struct WarmStart
    assignments::Dict{Int, Vector{ScheduleSegment}}
end

function _score_cell(flow::FlowSpec, times::Vector{Int}, coord::Tuple{Int, Int}, capacity_table)
    total = 0.0
    for t in times
        total += capacity_lookup(capacity_table, coord, t)
    end
    return total / (1 + travel_time(flow, coord))
end

function _order_cells(flow::FlowSpec, fc::FlowCandidate, capacity_table)
    scored = [
        (coord, _score_cell(flow, times, coord, capacity_table))
        for (coord, times) in fc.slots
    ]
    sort!(scored; by = x -> x[2], rev = true)
    return [item[1] for item in scored]
end

function build_warm_start(problem::ProblemSpec, candidates::CandidateIndex, capacity_table)
    remaining = Dict{Tuple{Int, Int}, Vector{Float64}}()
    for (coord, caps) in capacity_table
        remaining[coord] = copy(caps)
    end

    assignments = Dict{Int, Vector{ScheduleSegment}}()
    flows_sorted = sort(problem.flows; by = f -> f.start_time)

    for flow in flows_sorted
        fc = candidates.flows[flow.id]
        segments = ScheduleSegment[]
        assigned = 0.0
        occupied_times = Set{Int}()

        ordered_cells = _order_cells(flow, fc, remaining)
        for coord in ordered_cells
            times = sort(fc.slots[coord])
            for t in times
                if t in occupied_times
                    continue
                end
                caps = remaining[coord]
                avail = caps[t + 1]
                if avail <= EPS
                    continue
                end
                needed = flow.volume - assigned
                if needed <= EPS
                    break
                end
                take = min(needed, avail)
                if take <= EPS
                    continue
                end
                caps[t + 1] -= take
                push!(segments, ScheduleSegment(t, coord[1], coord[2], take))
                push!(occupied_times, t)
                assigned += take
            end
            if assigned + EPS >= flow.volume
                break
            end
        end

        assignments[flow.id] = segments
    end

    return WarmStart(assignments)
end

end # module
