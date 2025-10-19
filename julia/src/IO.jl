module IOUtil

export GridSpec, UAVSpec, FlowSpec, ProblemSpec, ScheduleSegment, FlowAssignment, ScoreSlice, ScoreSummary,
       read_problem, read_problem_from_string, load_problem, schedule_to_dict, schedule_to_text, write_output_json

using JSON3
using StructTypes

struct GridSpec
    rows::Int
    cols::Int
    horizon::Int
end

struct UAVSpec
    x::Int
    y::Int
    peak_bandwidth::Float64
    phase::Int
end

struct AccessCoord
    x::Int
    y::Int
end

struct ZoneSpec
    x_min::Int
    y_min::Int
    x_max::Int
    y_max::Int
end

struct FlowSpec
    id::Int
    access::AccessCoord
    start_time::Int
    volume::Float64
    zone::ZoneSpec
end

struct ProblemSpec
    grid::GridSpec
    uavs::Vector{UAVSpec}
    flows::Vector{FlowSpec}
end

StructTypes.StructType(::Type{GridSpec}) = StructTypes.Struct()
StructTypes.StructType(::Type{UAVSpec}) = StructTypes.Struct()
StructTypes.StructType(::Type{AccessCoord}) = StructTypes.Struct()
StructTypes.StructType(::Type{ZoneSpec}) = StructTypes.Struct()
StructTypes.StructType(::Type{FlowSpec}) = StructTypes.Struct()
StructTypes.StructType(::Type{ProblemSpec}) = StructTypes.Struct()

struct ScheduleSegment
    time::Int
    x::Int
    y::Int
    rate::Float64
end

struct FlowAssignment
    flow_id::Int
    segments::Vector{ScheduleSegment}
end

struct ScoreSlice
    flow_id::Int
    traffic::Float64
    delay::Float64
    distance::Float64
    landing::Float64
    weight::Float64
end

struct ScoreSummary
    total::Float64
    by_flow::Vector{ScoreSlice}
end

StructTypes.StructType(::Type{ScheduleSegment}) = StructTypes.Struct()
StructTypes.StructType(::Type{FlowAssignment}) = StructTypes.Struct()
StructTypes.StructType(::Type{ScoreSlice}) = StructTypes.Struct()
StructTypes.StructType(::Type{ScoreSummary}) = StructTypes.Struct()

function _build_problem(M::Int, N::Int, T::Int, uavs::Vector{UAVSpec}, flows::Vector{FlowSpec})
    grid = GridSpec(M, N, T)
    return ProblemSpec(grid, uavs, flows)
end

function _parse_tokens(tokens::Vector{SubString{String}})
    total = length(tokens)
    if total < 4
        error("Problem input is too short to contain grid header.")
    end
    idx = 1
    function next_token()
        if idx > total
            error("Unexpected end of input while parsing problem definition.")
        end
        tok = tokens[idx]
        idx += 1
        return tok
    end

    M = parse(Int, next_token())
    N = parse(Int, next_token())
    FN = parse(Int, next_token())
    T = parse(Int, next_token())

    cells = M * N
    uavs = Vector{UAVSpec}(undef, cells)
    for i in 1:cells
        x = parse(Int, next_token())
        y = parse(Int, next_token())
        B = parse(Float64, next_token())
        phase = parse(Int, next_token())
        uavs[i] = UAVSpec(x, y, B, phase)
    end

    flows = Vector{FlowSpec}(undef, FN)
    for i in 1:FN
        fid = parse(Int, next_token())
        ax = parse(Int, next_token())
        ay = parse(Int, next_token())
        t_start = parse(Int, next_token())
        volume = parse(Float64, next_token())
        m1 = parse(Int, next_token())
        n1 = parse(Int, next_token())
        m2 = parse(Int, next_token())
        n2 = parse(Int, next_token())
        access = AccessCoord(ax, ay)
        zone = ZoneSpec(m1, n1, m2, n2)
        flows[i] = FlowSpec(fid, access, t_start, volume, zone)
    end

    return _build_problem(M, N, T, uavs, flows)
end

function _parse_in_text(raw::AbstractString)
    tokens = split(raw)
    return _parse_tokens(tokens)
end

function _parse_json_text(raw::AbstractString)
    return JSON3.read(raw, ProblemSpec)
end

function read_problem_from_string(raw::AbstractString)
    trimmed = strip(raw)
    isempty(trimmed) && error("Input is empty; expected problem definition.")
    first_char = findfirst(!isspace, raw)
    if first_char === nothing
        error("Input is empty; expected problem definition.")
    end
    ch = raw[first_char]
    if ch == '{' || ch == '['
        return _parse_json_text(raw)
    else
        return _parse_in_text(raw)
    end
end

function read_problem(io::Base.IO)
    raw = read(io, String)
    return read_problem_from_string(raw)
end

function load_problem(path::AbstractString)
    ext = lowercase(get(splitext(path), 2, ""))
    if ext == ".json"
        open(path, "r") do f
            return _parse_json_text(read(f, String))
        end
    elseif ext == ".in" || ext == ""
        open(path, "r") do f
            return _parse_in_text(read(f, String))
        end
    else
        open(path, "r") do f
            return read_problem(f)
        end
    end
end

function schedule_to_dict(assignments::Vector{FlowAssignment})
    return Dict(
        :assignments => [
            Dict(
                :flow_id => fa.flow_id,
                :segments => [
                    Dict(:time => seg.time, :x => seg.x, :y => seg.y, :rate => seg.rate)
                    for seg in fa.segments
                ],
            )
            for fa in assignments
        ],
    )
end

function schedule_to_text(assignments::Vector{FlowAssignment})
    lines = IOBuffer()
    for fa in sort(assignments; by = x -> x.flow_id)
        segs = sort(fa.segments; by = x -> (x.time, x.x, x.y))
        println(lines, "$(fa.flow_id) $(length(segs))")
        for seg in segs
            println(lines, "$(seg.time) $(seg.x) $(seg.y) $(round(seg.rate; digits = 6))")
        end
    end
    return String(take!(lines))
end

function write_output_json(path::AbstractString, assignments::Vector{FlowAssignment}, score::ScoreSummary)
    data = Dict(
        :assignments => [
            Dict(
                :flow_id => fa.flow_id,
                :segments => [
                    Dict(:time => seg.time, :x => seg.x, :y => seg.y, :rate => seg.rate)
                    for seg in fa.segments
                ],
            )
            for fa in assignments
        ],
        :score => Dict(
            :total => score.total,
            :by_flow => [
                Dict(
                    :flow_id => fs.flow_id,
                    :traffic => fs.traffic,
                    :delay => fs.delay,
                    :distance => fs.distance,
                    :landing => fs.landing,
                    :weight => fs.weight,
                )
                for fs in score.by_flow
            ],
        ),
    )
    open(path, "w") do f
        JSON3.write(f, data; indent = 2)
    end
end

end # module
