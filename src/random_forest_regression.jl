# Random forests
using RDatasets
using Compat
using Distributions
using Gallium

include("measures.jl")
include("classification.jl")

const NO_BEST = (0, 0)

immutable Leaf
    majority::Any
    values::Vector
end

@compat immutable Node
    col_idx::Integer
    split_value::Any
    surrogates::Array{Tuple{Int, Real}, 1}                # This holds our surrogate vars

    # pointers to daughter nodes
    left::Union{Leaf, Node}
    right::Union{Leaf, Node}
end

@compat typealias LeafOrNode Union{Leaf, Node}



"""
Finds the threshold to split `features` with that minimizes the
mean-squared-error loss over `labels`.
Returns (best_val, best_thresh), where `best_val` is -MSE

Variable name changes from original code
  labels   -> y
  features -> x
  nl       -> n_left
  nr       -> n_right
  su       -> sum_y
  su2      -> sum_y2
  s_l      -> sum_y_left
  s2_l     -> sum_y2_left
  s_r      -> sum_y_right
  s2_r     -> sum_y2_right

"""
function _best_mse_loss{T<:Float64, U<:Real}(y::Vector{T}, x::Vector{U}, domain)
    best_val = -Inf
    best_thresh = 0.0
    n = length(y)

    sum_y_left = sum_y2_left = zero(T)            # scalar values of 0

    sum_y = sum(y)::T                             # scalar sum of all y_i
    sum_y2 = zero(T);                             # scalar value of 0

    # get sum of squares
    for i = 1:n
        sum_y2 += y[i]^2
    end

    n_left = 0
    i = 1

    # Since`x` is sorted, below is an O(n) algorithm for finding the optimal
    # threshold in `domain`. We iterate through the array and update sum_y_left
    # and sum_y_right (= sum(y) - sum_y_left) as we go. - @cstjean
    for thresh in domain

        # this loop checks which side of the split this x_i is on
        while i <= n && x[i] < thresh
            sum_y_left += y[i]
            sum_y2_left += y[i]^2
            n_left += 1
            i += 1
        end
        sum_y_right = sum_y - sum_y_left
        sum_y2_right = sum_y2 - sum_y2_left
        n_right = n - n_left

        # This check is necessary I think because in theory all y could
        # be the same, then either n_left or n_right would be 0. - @cstjean
        if n_right > 0 && n_left > 0

            # This isn't really squared-error loss. We're computing the sum of variance
            # in the daughter nodes times the N for that daughter node. This last point
            # is why the two paranthetical terms here look a bit diffent than mere variance.
            # We have canceled a constant term N for each term.
            loss = (sum_y2_left - sum_y_left^2/n_left) + (sum_y2_right - sum_y_right^2/n_right)

            # update best value and threshold
            if -loss > best_val
                best_val = -loss
                best_thresh = thresh
            end
        end
    end
    return best_val, best_thresh
end


"""
This function returns a tuple with the column index and the threshold
value of the predictor in the matrix X that minimizes MSE. The function
assumes complete data in both y and X.

Variable name changes from original code
  labels     -> y
  features   -> x
  nr         -> n
  nf         -> p
  i          -> j
  features_i -> x_j
  labels_i   -> y_ord
  domain_i   -> domain_j
  inds       -> col_indcs

"""
function _split_mse{T<:Float64, U<:Real}(y::Vector{T}, X::Matrix{U}, nsubfeatures::Int)
    n, p = size(X)
    best = NO_BEST
    best_val = -Inf

    if nsubfeatures > 0
        r = randperm(p)
        col_indcs = r[1:nsubfeatures]
    else
        col_indcs = 1:p
    end

    for j in col_indcs
        # Sorting used to be performed only when n <= 100, but doing it
        # unconditionally improved fitting performance by 20%. It's a bit of a
        # puzzle. Either it improved type-stability, or perhaps branch
        # prediction is much better on a sorted sequence.
        ord = sortperm(X[:, j])
        x_j = X[ord, j]
        y_ord = y[ord]

        if n > 100
            if VERSION >= v"0.4.0-dev"
                domain_j = quantile(x_j, linspace(0.01, 0.99, 99); sorted=true)
            else  # sorted=true isn't supported on StatsBase's Julia 0.3 version
                domain_j = quantile(x_j, linspace(0.01, 0.99, 99))
            end
        else
            domain_j = x_j
        end
        value, thresh = _best_mse_loss(y_ord, x_j, domain_j)

        if value > best_val
            best_val = value
            best = (j, thresh)
        end
    end
    return best
end




function _split_mse_df{T<:Float64}(y::Vector{T}, X::DataFrame, nsubfeatures::Int)
    n, p = size(X)
    best = NO_BEST
    best_val = -Inf
    if nsubfeatures > 0
        r = randperm(p)
        col_indcs = r[1:nsubfeatures]
    else
        col_indcs = 1:p
    end
    for j in col_indcs
        keep_row = !isna(X[:, j])
        x_obs = convert(Vector, X[keep_row, j])
        y_obs = y[keep_row]

        ord = sortperm(x_obs)
        x_j = convert(Array{typeof(x_obs[1])}, x_obs[ord])
        y_ord = y_obs[ord]

        if n > 100
            if VERSION >= v"0.4.0-dev"
                domain_j = quantile(x_j, linspace(0.01, 0.99, 99); sorted=true)
            else  # sorted=true isn't supported on StatsBase's Julia 0.3 version
                domain_j = quantile(x_j, linspace(0.01, 0.99, 99))
            end
        else
            domain_j = x_j
        end
        value, thresh = _best_mse_loss(y_ord, x_j, domain_j)
        if value > best_val
            best_val = value
            best = (j, thresh)
        end
    end
    return best
end



# d = dataset("datasets", "airquality")
# d[:row_idx] = 1:nrow(d)
# dc = d[complete_cases(d), :];
# X = convert(Array, dc[:, 2:6]);
# y = convert(Array{Float64,1}, dc[:, 1]);
#
# @time _split_mse(y, X, 0)
#
# keep_idx = !isna(d[:,1])
# @time _split_mse_df(convert(Vector{Float64}, d[keep_idx, 1]), d[keep_idx, 2:6], 0)










function find_na_cols(dat)
    cols_with_na = Array{Int, 1}(0)

    for j in 1:ncol(dat)
        if sum(isna(dat[:, j])) ≠ 0
            push!(cols_with_na, j)
        end
    end
    return cols_with_na
end


# function find_surrogates{T::BitArray}(left_node::T, y::Vector, X::Matrix, nsubfeatures::Int)



apply_tree(leaf::Leaf, feature::Vector) = leaf.majority

function apply_tree(tree::Node, features::Vector)
    if tree.split_value == nothing                  # when would this be true???
        return apply_tree(tree.left, features)
    elseif features[tree.col_idx] < tree.split_value
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end


# function apply_tree(tree::LeafOrNode, features::Matrix)
#     n = size(features,1)
#     predictions = Array(Any, n)
#     for i in 1:n
#         predictions[i] = apply_tree(tree, squeeze(features[i,:],1))
#     end
#     if typeof(predictions[1]) <: Float64
#         return float(predictions)
#     else
#         return predictions
#     end
# end


function build_stump{T<:Float64, U<:Real}(y::Vector{T}, X::Matrix{U})
    S = _split_mse(y, X, 0)

    if S == NO_BEST
        return Leaf(mean(y), y)
    end

    col_idx, thresh = S
    split = X[:, col_idx] .< thresh

    return Node(col_idx,
                thresh,
                Leaf(mean(y[split]), y[split]),
                Leaf(mean(y[!split]), y[!split]))
end



function surrogate_splits(y_obs_split::Vector, X::DataFrame, col_indcs::Vector{Int}, max_surrogates::Int, weights::Vector)
    surr = Array{Tuple}(max_surrogates)
    p = ncol(X)
    n_surr = p - 1 < max_surrogates ? p - 1: max_surrogates

    for i = 1:n_surr
        surr[i] = _split_classifcation_error_loss(y_obs_split, X, col_indcs, weights)
        col_indcs = setdiff(col_indcs, surr[i][1])
    end
    return surr
end

# This method is dispatched when weights are omitted. This
# allows us to compute the loss function 5x faster
function surrogate_splits(y_obs_split::Vector, X::DataFrame, row_indcs::Vector{Int}, col_indcs::Vector{Int}, max_surrogates::Int)
    surr = Array{Tuple}(max_surrogates)
    p = ncol(X)
    n_surr = p - 1 < max_surrogates ? p - 1: max_surrogates

    for i = 1:n_surr
        surr[i] = _split_classifcation_error_loss(y_obs_split, X, row_indcs, col_indcs)
        col_indcs = setdiff(col_indcs, surr[i][1])
    end
    return surr
end




function build_tree_df{T<:Float64}(y::Vector{T}, X::DataFrame, row_indcs, maxlabels=5, nsubfeatures=0, maxdepth=-1, max_surrogates=5)
    n = size(X, 1)
    if maxdepth < -1
        error("Unexpected value for maxdepth: $(maxdepth) (expected: maxdepth >= 0, or maxdepth = -1 for infinite depth)")
    end

    if length(y) <= maxlabels || maxdepth == 0          # stopping rules
        return Leaf(mean(y), y)
    end

    S = _split_mse_df(y, X, nsubfeatures)               # get [complete data] optimal split point

    if S == NO_BEST
        return Leaf(mean(y), y)
    end

    col_idx, thresh = S
    if max_surrogates ≠ 0
        cols_with_na = find_na_cols(X)                  # needed for all recursive steps since rows of X change
    end
    if col_idx in cols_with_na
        na_rows = isna(X[:, col_idx])
        split_with_na = Array{Any, 1}(n)                # vector of Bools with some NA values

        for i = 1:n
            split_with_na[i] = isna(X[i, col_idx]) ? NA : X[i, col_idx] < thresh
        end

        row_indcs = row_indcs[!na_rows]
        col_indcs = deleteat!(collect(1:ncol(X)), col_idx)
        X_2 = X[:, col_indcs]

        # Here we need a function that splits so as to optimize agreement
        # with the `split_with_na` result for each observed `row_indcs`.
        surrogate_vars = surrogate_splits(split_with_na, X_2, row_indcs, col_indcs, 5)


    else
        split = X[:, col_idx] .< thresh
    end

    return Node(col_idx,
                thresh,
                surrogate_vars,
                build_tree_df(y[split], X[split,:], row_indcs[split], maxlabels, nsubfeatures, max(maxdepth-1, -1)),
                build_tree_df(y[!split], X[!split,:], row_indcs[!split], maxlabels, nsubfeatures, max(maxdepth-1, -1)))
end

n = 10
p = 3
X = DataFrame(randn(n, p));
y = randn(n);
X_mis = add_missing(X, 0.3)
build_tree_df(y, X_mis, collect(1:n))























# Spare parts


# function make_na_indicator(dat)
#     X_hasna = BitArray{2}(nrow(dat), ncol(dat))
#
#     for j in 1:ncol(dat)
#         X_hasna[:, j] = isna(dat[:, j])
#     end
#     return X_hasna
# end

# function build_tree{T<:Float64}(y::Vector{T}, X::DataFrame, row_indcs, maxlabels=5, nsubfeatures=0, maxdepth=-1)
#
#     if maxdepth < -1
#         error("Unexpected value for maxdepth: $(maxdepth) (expected: maxdepth >= 0, or maxdepth = -1 for infinite depth)")
#     end
#
#     if length(y) <= maxlabels || maxdepth == 0            # stopping rules
#         return Leaf(mean(y), y)
#     end
#
#     S = _split_mse(y, X, nsubfeatures)
#
#     if S == NO_BEST
#         return Leaf(mean(y), y)
#     end
#
#     col_idx, thresh = S
#
#     split = X[:, col_idx] .< thresh
#     return Node(col_idx,
#                 thresh,
#                 build_tree(y[split], X[split,:], row_indcs[split], maxlabels, nsubfeatures, max(maxdepth-1, -1)),
#                 build_tree(y[!split], X[!split,:], row_indcs[!split], maxlabels, nsubfeatures, max(maxdepth-1, -1)))
# end
