


function surrogate_splits(y_obs_split::Vector, X::DataFrame, col_indcs::Vector{Int}, max_surrogates::Int, weights::Vector)
    p = ncol(X)
    n_surr = p - 1 < max_surrogates ? p - 1: max_surrogates
    surr = Array{Tuple}(n_surr)

    for i = 1:n_surr
        surr[i] = _split_classifcation_error_loss(y_obs_split, X, col_indcs, weights)
        col_indcs = setdiff(col_indcs, surr[i][1])
    end
    return surr
end

# This method is dispatched when weights are omitted. This
# allows us to compute the loss function 5x faster
function surrogate_splits(y_obs_split::Vector, X::DataFrame, row_indcs::Vector{Int}, col_indcs::Vector{Int}, max_surrogates::Int)
    p = ncol(X)
    n_surr = p - 1 < max_surrogates ? p - 1: max_surrogates
    surr = Array{Tuple}(n_surr)

    for i = 1:n_surr
        surr[i] = _split_classifcation_error_loss(y_obs_split, X, row_indcs, col_indcs)
        col_indcs = setdiff(col_indcs, surr[i][1])
    end
    return surr
end

# For now, this function doesn't solve the edge case in which
# a given record has missing data on all surrogates.
function apply_surrogates(split_with_na::Vector, X::DataFrame, surr::Array{Tuple, 1})
    n = length(split_with_na)
    display(surr)
    col_indcs = [x[1] for x in surr]
    warn("this is it")

    col_thresh = [x[2] for x in surr]

    split = falses(n)

    for i = 1:n
        warn("made it here at iteration $i")
        if isna(split_with_na[i])
            for (idx, j) in enumerate(col_indcs)
                if !isna(X[i, j])
                    split[i] = X[i, j] .< col_thresh[idx]
                    break
                end
            end
        end
        split[i] = split_with_na[i]
    end
    return split
end
