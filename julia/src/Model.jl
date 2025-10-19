module Model

export SolverConfig, ClusterSolution, solve_cluster

using JuMP
using HiGHS

const MOI = JuMP.MOI

using ..IOUtil: ProblemSpec, FlowSpec, ScheduleSegment
using ..Candidates: CandidateIndex, FlowCandidate, travel_time
using ..Clustering: Cluster
using ..Heuristics: WarmStart
using ..Capacity: capacity_lookup

const SlotKey = Tuple{Int, Tuple{Int, Int}, Int}
const EPS = 1e-6

struct SolverConfig
    total_time_limit::Float64
    cluster_time_floor::Float64
    slack_penalty::Float64
    landing_penalty_scale::Float64
    mip_gap::Float64
end

SolverConfig(; total_time_limit::Float64 = 5.0, cluster_time_floor::Float64 = 0.5, slack_penalty::Float64 = 1e5,
             landing_penalty_scale::Float64 = 1.0, mip_gap::Float64 = 0.01) =
    SolverConfig(total_time_limit, cluster_time_floor, slack_penalty, landing_penalty_scale, mip_gap)

struct ClusterSolution
    assignments::Dict{Int, Vector{ScheduleSegment}}
    status::MOI.TerminationStatusCode
    objective::Float64
end

function _flow_lookup(problem::ProblemSpec)
    return Dict(flow.id => flow for flow in problem.flows)
end

function _init_maps(cluster::Cluster)
    flow_to_x = Dict{Int, Vector{SlotKey}}()
    for fid in cluster.flow_ids
        flow_to_x[fid] = SlotKey[]
    end
    return flow_to_x
end

function _build_indices(cluster::Cluster, candidates::CandidateIndex)
    x_keys = SlotKey[]
    flow_time = Dict{Tuple{Int, Int}, Vector{SlotKey}}()
    cell_time = Dict{Tuple{Tuple{Int, Int}, Int}, Vector{SlotKey}}()
    flow_to_x = _init_maps(cluster)
    flow_cell = Dict{Tuple{Int, Tuple{Int, Int}}, Vector{SlotKey}}()
    y_keys = Set{Tuple{Int, Tuple{Int, Int}}}()

    for fid in cluster.flow_ids
        fc = candidates.flows[fid]
        for (cell, times) in fc.slots
            for t in times
                key = (fid, cell, t)
                push!(x_keys, key)
                push!(get!(flow_time, (fid, t), SlotKey[]), key)
                push!(get!(cell_time, (cell, t), SlotKey[]), key)
                push!(get!(flow_to_x, fid, SlotKey[]), key)
                push!(get!(flow_cell, (fid, cell), SlotKey[]), key)
                push!(y_keys, (fid, cell))
            end
        end
    end

    return x_keys, flow_time, cell_time, flow_to_x, flow_cell, collect(y_keys)
end

function _x_coeff(flow::FlowSpec, cell::Tuple{Int, Int}, t::Int)
    volume_term = 100.0 * 0.4 / flow.volume
    T_max = 10.0
    delay_term = 100.0 * 0.2 * (T_max / (t + T_max)) / flow.volume
    alpha = 0.1
    distance = travel_time(flow, cell)
    distance_term = 100.0 * 0.3 * (2.0 ^ (-alpha * distance)) / flow.volume
    return volume_term + delay_term + distance_term
end

function _landing_penalty(cfg::SolverConfig, fc::FlowCandidate, cell::Tuple{Int, Int})
    base = 100.0 * 0.1 / max(1, length(fc.slots))
    return cfg.landing_penalty_scale * base
end

function _apply_warm_start!(warm_start::WarmStart, x, u, y, drop, cluster::Cluster, flow_lookup, x_keys_set::Set{SlotKey},
                            y_keys_set::Set{Tuple{Int, Tuple{Int, Int}}})
    for fid in cluster.flow_ids
        flow = flow_lookup[fid]
        assigned = 0.0
        segments = get(warm_start.assignments, fid, ScheduleSegment[])
        for seg in segments
            key = (fid, (seg.x, seg.y), seg.time)
            if key in x_keys_set
                JuMP.set_start_value(x[key], seg.rate)
                assigned += seg.rate
                if key in x_keys_set
                    JuMP.set_start_value(u[key], 1.0)
                end
                y_key = (fid, (seg.x, seg.y))
                if y_key in y_keys_set
                    JuMP.set_start_value(y[y_key], 1.0)
                end
            end
        end
        JuMP.set_start_value(drop[fid], max(0.0, flow.volume - assigned))
    end
end

function _extract_solution(x, cluster::Cluster, x_keys::Vector{SlotKey})
    assignments = Dict{Int, Vector{ScheduleSegment}}()
    for fid in cluster.flow_ids
        assignments[fid] = ScheduleSegment[]
    end
    for key in x_keys
        value = JuMP.value(x[key])
        if value > EPS
            fid, cell, t = key
            push!(assignments[fid], ScheduleSegment(t, cell[1], cell[2], value))
        end
    end
    for fid in cluster.flow_ids
        sort!(assignments[fid]; by = s -> s.time)
    end
    return assignments
end

function _fallback_from_warmstart(warm_start::WarmStart, cluster::Cluster)
    assignments = Dict{Int, Vector{ScheduleSegment}}()
    for fid in cluster.flow_ids
        segs = get(warm_start.assignments, fid, ScheduleSegment[])
        assignments[fid] = copy(segs)
    end
    return assignments
end

function solve_cluster(cluster::Cluster, problem::ProblemSpec, candidates::CandidateIndex, capacity_table,
                       cfg::SolverConfig, warm_start::WarmStart, requested_time::Float64)
    flow_lookup = _flow_lookup(problem)
    x_keys, flow_time, cell_time, flow_to_x, flow_cell, y_keys_vec = _build_indices(cluster, candidates)
    y_keys_set = Set(y_keys_vec)

    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    limit = max(cfg.cluster_time_floor, requested_time)
    limit = min(limit, cfg.total_time_limit)
    if limit > 0
        MOI.set(model, MOI.TimeLimitSec(), limit)
    end
    if cfg.mip_gap > 0
        JuMP.set_optimizer_attribute(model, "mip_rel_gap", cfg.mip_gap)
    end

    x = @variable(model, [k in x_keys], lower_bound = 0.0)
    u = @variable(model, [k in x_keys], Bin)
    y = @variable(model, [k in y_keys_vec], Bin)
    drop = @variable(model, [fid in cluster.flow_ids], lower_bound = 0.0)

    for key in x_keys
        fid, cell, t = key
        cap = capacity_lookup(capacity_table, cell, t)
        @constraint(model, x[key] <= cap * u[key])
    end

    for (ft, keys) in flow_time
        @constraint(model, sum(u[key] for key in keys) <= 1)
    end

    for fid in cluster.flow_ids
        coeffs = get(flow_to_x, fid, SlotKey[])
        expr = isempty(coeffs) ? 0.0 : sum(x[key] for key in coeffs)
        flow = flow_lookup[fid]
        @constraint(model, expr + drop[fid] == flow.volume)
    end

    for ((fid, cell), keys) in flow_cell
        if isempty(keys)
            continue
        end
        y_key = (fid, cell)
        if y_key in y_keys_set
            @constraint(model, y[y_key] <= sum(u[key] for key in keys))
            for key in keys
                @constraint(model, u[key] <= y[y_key])
            end
        end
    end

    for ((cell, t), keys) in cell_time
        cap = capacity_lookup(capacity_table, cell, t)
        @constraint(model, sum(x[key] for key in keys) <= cap)
    end

    coeffs = Dict{SlotKey, Float64}()
    for key in x_keys
        fid, cell, t = key
        flow = flow_lookup[fid]
        coeffs[key] = _x_coeff(flow, cell, t)
    end

    landing_penalty = Dict{Tuple{Int, Tuple{Int, Int}}, Float64}()
    for y_key in y_keys_vec
        fid, cell = y_key
        fc = candidates.flows[fid]
        landing_penalty[y_key] = _landing_penalty(cfg, fc, cell)
    end

    obj_expr = sum(coeffs[key] * x[key] for key in x_keys)
    if !isempty(y_keys_vec)
        obj_expr -= sum(landing_penalty[y_key] * y[y_key] for y_key in y_keys_vec)
    end
    obj_expr -= sum(cfg.slack_penalty * drop[fid] for fid in cluster.flow_ids)
    @objective(model, Max, obj_expr)

    if warm_start !== nothing
        x_keys_set = Set(x_keys)
        _apply_warm_start!(warm_start, x, u, y, drop, cluster, flow_lookup, x_keys_set, y_keys_set)
    end

    optimize!(model)
    status = JuMP.termination_status(model)

    if JuMP.result_count(model) == 0
        assignments = if warm_start === nothing
            Dict(fid => ScheduleSegment[] for fid in cluster.flow_ids)
        else
            _fallback_from_warmstart(warm_start, cluster)
        end
        return ClusterSolution(assignments, status, 0.0)
    end

    assignments = _extract_solution(x, cluster, x_keys)
    obj = try
        JuMP.objective_value(model)
    catch
        0.0
    end
    return ClusterSolution(assignments, status, obj)
end

end # module
