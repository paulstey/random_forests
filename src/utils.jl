import Base.length

length(leaf::Leaf) = 1
length(tree::Node) = length(tree.left) + length(tree.right)
length(ensemble::Ensemble) = length(ensemble.trees)


depth(leaf::Leaf) = 0
depth(tree::Node) = 1 + max(depth(tree.left), depth(tree.right))


# Functions for determining the predictors used in a tree

function get_predictors_used!(tree, col_indcs)
    if isa(tree, Leaf)
        return nothing 
    else 
        push!(col_indcs, tree.col_idx)
        get_predictors_used!(tree.left, col_indcs)
        get_predictors_used!(tree.right, col_indcs)
    end 
end 

# Given a tree, this function returns the indices of the 
# columns used at each node. Note that surrogates are ignored.
function predictors_used(tree)
    col_indcs = Array{Int}(0)
    get_predictors_used!(tree, col_indcs)
    return unique(col_indcs) 
end 


function add_missing(dat::DataFrame, pr)
    d = Bernoulli(pr)
    X = copy(dat)
    n, p = size(X)
    for j = 1:p
        X[:, j] = convert(DataArray{Any, 1}, X[:, j])
        for i = 1:n
            if rand(d) == 1
                X[i, j] = NA
            end
        end
    end
    return X
end


function mean_impute(v)
    μ = mean(dropna(v))
    n = length(v)
    res = zeros(n)
    for i = 1:n
        if isna(v[i])
            res[i] = μ
        else
            res[i] = v[i]
        end
    end
    res
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
