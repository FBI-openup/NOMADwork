using Test
using NTNScheduler
import NTNScheduler
const Capacity = NTNScheduler.Capacity
const Candidates = NTNScheduler.Candidates
const Heuristics = NTNScheduler.Heuristics
const PostProcess = NTNScheduler.PostProcess
const SolverMain = NTNScheduler.Main

const DATA_DIR = joinpath(@__DIR__, "data")
const EXAMPLE_JSON = joinpath(DATA_DIR, "example1.json")
const EXAMPLE_IN = joinpath(dirname(@__DIR__), "..", "examples", "example1.in")
const EXAMPLE_OUT = joinpath(dirname(@__DIR__), "..", "examples", "example1.out")

function load_example_problem()
    return NTNScheduler.IOUtil.load_problem(EXAMPLE_JSON)
end

@testset "Input parsing" begin
    json_problem = load_example_problem()
    in_problem = NTNScheduler.IOUtil.load_problem(EXAMPLE_IN)
    @test in_problem.grid.rows == json_problem.grid.rows
    @test in_problem.grid.cols == json_problem.grid.cols
    @test in_problem.grid.horizon == json_problem.grid.horizon
    @test length(in_problem.uavs) == length(json_problem.uavs)
    @test length(in_problem.flows) == length(json_problem.flows)
    raw_in = read(EXAMPLE_IN, String)
    parsed_from_string = NTNScheduler.IOUtil.read_problem_from_string(raw_in)
    @test parsed_from_string.grid.horizon == json_problem.grid.horizon
    @test parsed_from_string.flows[1].volume == json_problem.flows[1].volume
end

@testset "Capacity pattern" begin
    problem = load_example_problem()
    table = Capacity.build_capacity_table(problem)
    cap0 = Capacity.capacity_lookup(table, (0, 0), 0)
    cap4 = Capacity.capacity_lookup(table, (0, 0), 4)
    cap7 = Capacity.capacity_lookup(table, (0, 0), 7)
    @test isapprox(cap0, 10.0; atol = 1e-6)
    @test isapprox(cap4, 5.0; atol = 1e-6)
    @test isapprox(cap7, 0.0; atol = 1e-6)
end

@testset "Candidate generation" begin
    problem = load_example_problem()
    table = Capacity.build_capacity_table(problem)
    candidates = Candidates.build_candidates(problem, table, Candidates.CandidateConfig(base_window = 5))
    flow_lookup = Dict(flow.id => flow for flow in problem.flows)
    flow1 = flow_lookup[1]
    fc1 = candidates.flows[1]
    times = fc1.slots[(0, 0)]
    @test minimum(times) == flow1.start_time
end

@testset "Heuristic warm start respects capacity" begin
    problem = load_example_problem()
    table = Capacity.build_capacity_table(problem)
    candidates = Candidates.build_candidates(problem, table, Candidates.CandidateConfig())
    warm = Heuristics.build_warm_start(problem, candidates, table)
    usage = Dict{Tuple{Int, Tuple{Int, Int}}, Float64}()
    for segs in values(warm.assignments)
        for seg in segs
            key = (seg.time, (seg.x, seg.y))
            usage[key] = get(usage, key, 0.0) + seg.rate
        end
    end
    for (t_cell, total) in usage
        t, cell = t_cell
        @test total <= Capacity.capacity_lookup(table, cell, t) + 1e-6
    end
end

function load_schedule_segments(path::AbstractString)
    lines = readlines(path)
    assignments = Dict{Int, Vector{NTNScheduler.IOUtil.ScheduleSegment}}()
    i = 1
    while i <= length(lines)
        header = split(lines[i])
        flow_id = parse(Int, header[1])
        count = parse(Int, header[2])
        i += 1
        segs = NTNScheduler.IOUtil.ScheduleSegment[]
        for _ in 1:count
            t_str, x_str, y_str, rate_str = split(lines[i])
            push!(segs, NTNScheduler.IOUtil.ScheduleSegment(parse(Int, t_str), parse(Int, x_str), parse(Int, y_str), parse(Float64, rate_str)))
            i += 1
        end
        assignments[flow_id] = segs
    end
    return assignments
end

@testset "PostProcess scoring matches reference" begin
    problem = load_example_problem()
    assignments = load_schedule_segments(EXAMPLE_OUT)
    score = PostProcess.compute_score(problem, assignments)
    @test isapprox(score.total, 97.047; atol = 1e-3)
end

@testset "End-to-end solve" begin
    problem = load_example_problem()
    artifacts = SolverMain.solve_problem(problem)
    @test length(artifacts.assignments) == length(problem.flows)
    for flow in problem.flows
        @test haskey(artifacts.assignments, flow.id)
    end
    @test artifacts.score.total >= 0.0
end
