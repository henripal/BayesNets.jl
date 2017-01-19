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
    name_to_index::Dict{NodeName, Int}
    )
    
    # Dict already has unique node names
    ug = UG(length(name_to_index))

    # Build the UG by linking all edges within a given factor
    for factor in factors
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
end
MRF() = MRF(UG(0), Factor[], NodeName[], Dict{NodeName, Int}())
MRF{T <: Factor}(::Type{T}) = MRF(UG(0), T[], NodeName[], Dict{NodeName, Int}())

function MRF{T <: Factor}(factors::AbstractVector{T})
    name_to_index = Dict{NodeName, Int}()
    # We need a collection of unique nodes to create the graph
    names = unique(collect(Base.flatten([factor.dimensions for factor in factors])))

    for (i, node) in enumerate(names)
        name_to_index[node] = i
    end

    ug = _build_ug_from_factors(factors, name_to_index)

    MRF(ug, factors, names, name_to_index)
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

function has_edge(mrf::MRF, source::NodeName, target::NodeName)::Bool
    u = get(mrf.name_to_index, source, 0)
    v = get(mrf.name_to_index, target, 0)
    u != 0 && v != 0 && has_edge(mrf.ug, u, v)
end
   
