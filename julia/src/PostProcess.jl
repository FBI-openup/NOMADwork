module PostProcess

export merge_assignments!, normalize_assignments!, compute_score, assignments_to_flow_payload

using ..IOUtil: ProblemSpec, FlowSpec, ScheduleSegment, FlowAssignment, ScoreSlice, ScoreSummary
using ..Capacity: capacity_lookup

const EPS = 1e-6

function merge_assignments!(dest::Dict{Int, Vector{ScheduleSegment}}, src::Dict{Int, Vector{ScheduleSegment}})
    for (fid, segs) in src
        vec = get!(dest, fid, ScheduleSegment[])
        append!(vec, segs)
    end
    return dest
end

function normalize_assignments!(assignments::Dict{Int, Vector{ScheduleSegment}})
    for segs in values(assignments)
        sort!(segs; by = s -> (s.time, s.x, s.y))
    end
    return assignments
end

function _group_by_time(segments::Vector{ScheduleSegment})
    grouped = Dict{Int, Vector{ScheduleSegment}}()
    for seg in segments
        push!(get!(grouped, seg.time, ScheduleSegment[]), seg)
    end
    return grouped
end

function _score_flow(flow::FlowSpec, segments::Vector{ScheduleSegment})
    grouped = _group_by_time(segments)
    total_sent = sum(seg.rate for seg in segments)
    traffic_score = flow.volume > 0 ? min(1.0, total_sent / flow.volume) : 0.0

    T_max = 10.0
    delay_score = 0.0
    for (t, arr) in grouped
        q_sum = sum(seg.rate for seg in arr)
        delay_score += (T_max / (t + T_max)) * (q_sum / flow.volume)
    end

    alpha = 0.1
    distance_score = 0.0
    for (_, arr) in grouped
        for seg in arr
            dist = abs(flow.access.x - seg.x) + abs(flow.access.y - seg.y)
            distance_score += (seg.rate / flow.volume) * (2.0 ^ (-alpha * dist))
        end
    end

    times_sorted = sort(collect(keys(grouped)))
    unique_landings = Tuple{Int, Int}[]
    last = nothing
    for t in times_sorted
        arr = grouped[t]
        sorted_arr = sort(arr; by = s -> s.rate, rev = true)
        rep = (sorted_arr[1].x, sorted_arr[1].y)
        if last === nothing || rep != last
            push!(unique_landings, rep)
            last = rep
        end
    end
    k = max(1, length(unique_landings))
    landing_score = 1.0 / k

    flow_score = 100.0 * (
        0.4 * traffic_score + 0.2 * delay_score + 0.3 * distance_score + 0.1 * landing_score
    )

    return flow_score, traffic_score, delay_score, distance_score, landing_score
end

function compute_score(problem::ProblemSpec, assignments::Dict{Int, Vector{ScheduleSegment}})
    total_weight = sum(flow.volume for flow in problem.flows)
    if total_weight <= EPS
        return ScoreSummary(0.0, ScoreSlice[])
    end

    slices = ScoreSlice[]
    total_score = 0.0
    for flow in problem.flows
        segs = get(assignments, flow.id, ScheduleSegment[])
        if isempty(segs)
            push!(slices, ScoreSlice(flow.id, 0.0, 0.0, 0.0, 0.0, flow.volume / total_weight))
            continue
        end
        flow_score, traffic, delay, distance, landing = _score_flow(flow, segs)
        weight = flow.volume / total_weight
        total_score += weight * flow_score
        push!(slices, ScoreSlice(flow.id, traffic, delay, distance, landing, weight))
    end

    return ScoreSummary(total_score, slices)
end

function assignments_to_flow_payload(assignments::Dict{Int, Vector{ScheduleSegment}})
    return [
        FlowAssignment(fid, segs)
        for (fid, segs) in sort(collect(assignments); by = x -> x[1])
    ]
end

end # module
