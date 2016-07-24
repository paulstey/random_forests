import Base.convert 
import Base.length



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
    return Node(0, nothing, x, Leaf(nothing,[nothing]))
end 
promote_rule(::Type{Node}, ::Type{Leaf}) = Node
promote_rule(::Type{Leaf}, ::Type{Node}) = Node

length(leaf::Leaf) = 1
length(tree::Node) = length(tree.left) + length(tree.right)
length(ensemble::Ensemble) = length(ensemble.trees)


depth(leaf::Leaf) = 0
depth(tree::Node) = 1 + max(depth(tree.left), depth(tree.right))



function apply_forest(forest::Ensemble, X::Matrix)
    n = size(X, 1)
    y_hat = Array(Any, n)
    for i in 1:n
        if VERSION < v"0.5.0-dev"
            y_hat[i] = apply_forest(forest, squeeze(X[i, :], 1))
        else 
            y_hat[i] = apply_forest(forest, X[i, :])
        end 
    end
    if eltype(y_hat) <: Float64
        return float(y_hat)
    else
        return y_hat
    end
end


function apply_forest(forest::Ensemble, x::Vector)
    ntrees = length(forest)
    votes = Array(Any, ntrees)
    
    for i in 1:ntrees
        votes[i] = apply_tree(forest.trees[i], x)
    end
    
    if eltype(votes) <: Float64
        return mean(votes)
    else
        return majority_vote(votes)
    end
end



function print_tree(leaf::Leaf, depth=-1, indent=0)
    matches = find(leaf.values .== leaf.majority)
    ratio = string(length(matches)) * "/" * string(length(leaf.values))
    println("$(leaf.majority) : $(ratio)")
end

function print_tree(tree::Node, depth=-1, indent=0)
    if depth == indent
        println()
        return
    end
    println("Feature $(tree.col_idx), Threshold $(tree.split_value)")
    print("    " ^ indent * "L-> ")
    print_tree(tree.left, depth, indent + 1)
    print("    " ^ indent * "R-> ")
    print_tree(tree.right, depth, indent + 1)
end

function show(io::IO, leaf::Leaf)
    println(io, "Decision Leaf")
    println(io, "Majority: $(leaf.majority)")
    print(io,   "Samples:  $(length(leaf.values))")
end

function show(io::IO, tree::Node)
    println(io, "Decision Tree")
    println(io, "Leaves: $(length(tree))")
    print(io,   "Depth:  $(depth(tree))")
end

function show(io::IO, ensemble::Ensemble)
    println(io, "Ensemble of Decision Trees")
    println(io, "Trees:      $(length(ensemble))")
    println(io, "Avg Leaves: $(mean([length(tree) for tree in ensemble.trees]))")
    print(io,   "Avg Depth:  $(mean([depth(tree) for tree in ensemble.trees]))")
end

