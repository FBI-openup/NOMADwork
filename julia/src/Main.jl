module Main

export solve_problem, run_cli, CLIOptions, SolverArtifacts

using ..IOUtil
using ..Capacity
using ..Candidates
using ..Clustering
using ..Heuristics
using ..Model
using ..PostProcess

struct CLIOptions
    input_path::Union{Nothing, String}
    output_json::Union{Nothing, String}
    output_text::Union{Nothing, String}
    time_limit::Union{Nothing, Float64}
end

struct SolverArtifacts
    assignments::Dict{Int, Vector{IOUtil.ScheduleSegment}}
    score::IOUtil.ScoreSummary
    clusters::Vector{Clustering.Cluster}
end

function _parse_cli(args::Vector{String})
    input_path = nothing
    output_json = nothing
    output_text = nothing
    time_limit = nothing
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--input" || arg == "-i"
            i += 1
            input_path = args[i]
        elseif arg == "--output-json"
            i += 1
            output_json = args[i]
        elseif arg == "--output-text"
            i += 1
            output_text = args[i]
        elseif arg == "--time-limit"
            i += 1
            time_limit = parse(Float64, args[i])
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    return CLIOptions(input_path, output_json, output_text, time_limit)
end

function solve_problem(problem::IOUtil.ProblemSpec; solver_cfg = Model.SolverConfig(), candidate_cfg = Candidates.CandidateConfig())
    capacity_table = Capacity.build_capacity_table(problem)
    candidates = Candidates.build_candidates(problem, capacity_table, candidate_cfg)
    clusters = Clustering.cluster_flows(candidates)

    warm_start = Heuristics.build_warm_start(problem, candidates, capacity_table)

    assignments = Dict{Int, Vector{IOUtil.ScheduleSegment}}()
    total_flows = length(problem.flows)
    remaining = solver_cfg.total_time_limit

    for cluster in clusters
        share = total_flows > 0 ? length(cluster.flow_ids) / total_flows : 0.0
        cluster_limit = max(solver_cfg.cluster_time_floor, share * solver_cfg.total_time_limit)
        if remaining > 0
            cluster_limit = min(cluster_limit, remaining)
            remaining = max(0.0, remaining - cluster_limit)
        else
            cluster_limit = solver_cfg.cluster_time_floor
        end

        solution = Model.solve_cluster(cluster, problem, candidates, capacity_table, solver_cfg, warm_start, cluster_limit)
        PostProcess.merge_assignments!(assignments, solution.assignments)
    end

    PostProcess.normalize_assignments!(assignments)
    score = PostProcess.compute_score(problem, assignments)
    return SolverArtifacts(assignments, score, clusters)
end

default_config = Model.SolverConfig()

function run_cli(args::Vector{String})
    opts = _parse_cli(args)
    problem = if opts.input_path === nothing
        IOUtil.read_problem(stdin)
    else
        IOUtil.load_problem(opts.input_path)
    end
    solver_cfg = opts.time_limit === nothing ? default_config : Model.SolverConfig(total_time_limit = opts.time_limit)
    artifacts = solve_problem(problem; solver_cfg = solver_cfg)

    flow_payload = PostProcess.assignments_to_flow_payload(artifacts.assignments)

    if opts.output_json !== nothing
        IOUtil.write_output_json(opts.output_json, flow_payload, artifacts.score)
    end

    schedule_text = IOUtil.schedule_to_text(flow_payload)
    if opts.output_text === nothing
        print(schedule_text)
    else
        open(opts.output_text, "w") do f
            write(f, schedule_text)
        end
    end

    println(stderr, "Solved $(length(problem.flows)) flows | score=$(round(artifacts.score.total; digits=3))")
end

end # module
