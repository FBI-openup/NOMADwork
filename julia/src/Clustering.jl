module Clustering

export Cluster, cluster_flows

using ..Candidates: CandidateIndex

struct Cluster
    flow_ids::Vector{Int}
    landing_cells::Vector{Tuple{Int, Int}}
end

function cluster_flows(candidates::CandidateIndex)
    cell_to_flows = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (fid, fc) in candidates.flows
        for cell in keys(fc.slots)
            push!(get!(cell_to_flows, cell, Int[]), fid)
        end
    end

    visited = Set{Int}()
    clusters = Vector{Cluster}()

    for fid in keys(candidates.flows)
        if fid in visited
            continue
        end
        queue = [fid]
        idx = 1
        flow_ids = Int[]
        landing_cells = Set{Tuple{Int, Int}}()

        while idx <= length(queue)
            current = queue[idx]
            idx += 1
            if current in visited
                continue
            end
            push!(visited, current)
            push!(flow_ids, current)

            fc = candidates.flows[current]
            for cell in keys(fc.slots)
                push!(landing_cells, cell)
                for neighbour in get(cell_to_flows, cell, Int[])
                    if neighbour ∉ visited
                        push!(queue, neighbour)
                    end
                end
            end
        end

        sort!(flow_ids)
        lc_vec = sort(collect(landing_cells); by = x -> (x[1], x[2]))
        push!(clusters, Cluster(flow_ids, lc_vec))
    end

    sort!(clusters; by = c -> length(c.flow_ids), rev = true)
    return clusters
end

end # module
