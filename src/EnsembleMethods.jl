# EnsembleMethods
using Compat
using Distributions
using DataFrames
using Base.Threads

import Base.convert




const NO_BEST = (0, 0)

immutable Leaf
    majority::Any
    values::Vector
end

@compat immutable Node
    col_idx::Int
    split_value::Any
    surrogates::Array{Tuple{Int, Real}, 1}                # This holds our surrogate vars

    # pointers to daughter nodes
    left::Union{Leaf, Node}
    right::Union{Leaf, Node}
end

@compat typealias LeafOrNode Union{Leaf, Node}

immutable Ensemble
    trees::Vector{Node}
end


function convert(::Type{Node}, x::Leaf)
    warn("converting leaf to node")
    return Node(0, nothing, Array{Tuple{Int, Real}, 1}(5), x, Leaf(nothing,[nothing]))
end

promote_rule(::Type{Node}, ::Type{Leaf}) = Node
promote_rule(::Type{Leaf}, ::Type{Node}) = Node


include("utils.jl")
include("measures.jl")
include("classification.jl")
include("surrogates.jl")
include("apply_models.jl")
include("random_forest_regression.jl")
