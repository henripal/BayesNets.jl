#=
A Markov Random Field  (MRF) represents a probability distribution
over a set of variables, P(x₁, x₂, ..., xₙ)
It leverages relations between variables in order to efficiently encode the joint distribution.
A MRF is defined by an undirected graph in which each node is a variable
and contains an associated conditional probability distribution P(xⱼ | parents(xⱼ))
=#

typealias UG Graph
function _build_ug_from_factors{T<:Factor}(
    factors::AbstractVector{T},
    name_to_index::Dict{NodeName, Int},
    name_to_factor_indices::Dict{NodeName, Array{Int}}
    )
    
    # Dict already has unique node names
    ug = UG(length(name_to_index))

    # Build the UG by linking all edges within a given factor
    for (factor_index, factor) in enumerate(factors)
        for d in factor.dimensions
            if ~ (factor_index in name_to_factor_indices[d])
                push!(name_to_factor_indices[d], factor_index)
            end
        end
        
        for d1 in factor.dimensions, d2 in factor.dimensions        
            i, j = name_to_index[d1], name_to_index[d2]
            if i <j
                add_edge!(ug, i, j)
            end
        end
    end

    ug
end

type MRF{T<:Factor}
    ug::UG 
    factors::Vector{T} # the factors associated with the MRF
    names::Vector{NodeName}
    name_to_index::Dict{NodeName,Int} # NodeName → index in ug 
    name_to_factor_indices::Dict{NodeName, Array{Int}}
end
MRF() = MRF(UG(0), Factor[], NodeName[], Dict{NodeName, Int}(), Dict{NodeName, Array{Int}}())
MRF{T <: Factor}(::Type{T}) = MRF(UG(0), T[], NodeName[], Dict{NodeName, Int}(), Dict{NodeName, Array{Int}}())

function MRF{T <: Factor}(factors::AbstractVector{T})
    name_to_index = Dict{NodeName, Int}()
    name_to_factor_indices = Dict{NodeName, Array{Int}}()
    names = Array{Symbol}[]
    # We need a collection of unique nodes to create the graph
    if isempty(names)
        names = unique(collect(Base.flatten([factor.dimensions for factor in factors])))
    end

    for (i, node) in enumerate(names)
        name_to_index[node] = i
        name_to_factor_indices[node] = []
    end

    ug = _build_ug_from_factors(factors, name_to_index, name_to_factor_indices)

    MRF(ug, factors, names, name_to_index, name_to_factor_indices)
end

Base.get(mrf::MRF, i::Int) = mrf.names[i]
Base.length(mrf::MRF) = length(mrf.name_to_index)

"""
Returns the list of NodeNames
"""
function Base.names(mrf::MRF)
    retval = Array(NodeName, length(mrf)) 
    for (key,val) in mrf.name_to_index
        retval[val] = key
    end
    retval
end
    
"""
Returns the neighbors as a list of NodeNames
"""
function neighbors(mrf::MRF, target::NodeName)
    i = mrf.name_to_index[target]
    NodeName[mrf.names[j] for j in neighbors(mrf.ug, i)]
end

"""
Returns the markov blanket - here same as neighbors
"""
function markov_blanket(mrf::MRF, target::NodeName)
    return neighbors(mrf, target)
end

"""
Whether the MRF contains the given edge
"""
function has_edge(mrf::MRF, source::NodeName, target::NodeName)::Bool
    u = get(mrf.name_to_index, source, 0)
    v = get(mrf.name_to_index, target, 0)
    u != 0 && v != 0 && has_edge(mrf.ug, u, v)
end
   
"""
Returns whether the set of node names `x` is d-separated
from the set `y` given the set `given`
"""
function is_independent(mrf::MRF, x::AbstractVector{NodeName}, y::AbstractVector{NodeName}, given::AbstractVector{NodeName})
    ug_copy = copy(mrf.ug)
    # we copy the mrf, then remove all edges
    # from `given`; then calc the connected components
    # if x and y are in different connected components then they are independent

    x_index = [mrf.name_to_index[node] for node in x]
    y_index = [mrf.name_to_index[node] for node in y]
    g_index = [mrf.name_to_index[node] for node in given]

    for g in g_index
        for n in neighbors(mrf.ug, g)
            rem_edge!(ug_copy, g, n)
        end
    end

    conn_components = connected_components(ug_copy)

    for component in conn_components
        if !isempty(intersect(component, x_index))
            if !isempty(intersect(component, y_index))
                return false
            end
        end
    end

    return true
end

#### IO to be reworked.


function plot(mrf::MRF)
    plot(mrf.ug, AbstractString[string(s) for s in mrf.names])
end


@compat function Base.show(f::IO, a::MIME"image/svg+xml", mrf::MRF)
    show(f, a, plot(mrf))
end
