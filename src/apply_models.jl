# Apply fitted trees or forests

function apply_tree(leaf::Leaf, x::DataArray)
    if leaf.majority == nothing 
        error("leaf.majority == nothing for $x")
    end 
    return leaf.majority 
end 


function apply_tree(tree::Node, x::DataArray)
    if tree.split_value == nothing                  # true at leaf node, maybe??
        error("WOW!!! Actually made it where tree.split_value == nothing!!!")
        return apply_tree(tree.left, x)
    elseif isna(x[tree.col_idx])
        n_surr = count_surrogates(tree)
        
        if n_surr > 0               # most often, we do have surrogates
            
            for i = 1:n_surr
                idx, val = tree.surrogates[i]
                if idx ≠ 0 && !isna(x[idx])

                    if x[idx] < val
                        return apply_tree(tree.left, x)
                    else
                        return apply_tree(tree.right, x)
                    end
                # If there are no surrogates for this predictor we flip a
                # coin to decide left/right.
                elseif idx == 0 || (isna(x[idx]) && i == n_surr)
                    go_left = bitrand(1)[1]
                    if go_left
                        return apply_tree(tree.left, x)
                    else
                        return apply_tree(tree.right, x)
                    end
                end
            end 
        # There are corner cases where no initialized surrogates exist 
        elseif n_surr == 0 
            warn("No surrogates, so we flip a coin for observation: $x")
            go_left = bitrand(1)[1]
            if go_left
                return apply_tree(tree.left, x)
            else 
                return apply_tree(tree.right, x)
            end 
        end 
    # in the simplest case, we have observed value
    elseif x[tree.col_idx] < tree.split_value
        return apply_tree(tree.left, x)
    else
        return apply_tree(tree.right, x)
    end
end


function apply_tree(tree::LeafOrNode, X::DataFrame)
    n = size(X, 1)
    predictions = Array(Any, n)             # this should not be type Any
    for i in 1:n
        # Careful here: conversion to DataArray does not drop dimension,
        # so X is a 1-row matrix; but this this works for now.
        predictions[i] = apply_tree(tree, DataArray(X[i, :])')
        if predictions[i] == nothing 
            error("Prediction for row $i == nothing")
        end 
    end
    return predictions
end


# Functions for applying the fitted random forest

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

    if eltype(votes) <: AbstractFloat
        return mean(votes)
    else
        return majority_vote(votes)
    end
end
