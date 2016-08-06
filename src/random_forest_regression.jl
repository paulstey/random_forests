# Random forests regression


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
# function _split_mse{T<:Float64, U<:Real}(y::Vector{T}, X::Matrix{U}, mtry::Int)
#     n, p = size(X)
#     best = NO_BEST
#     best_val = -Inf

#     if mtry > 0
#         r = randperm(p)
#         col_indcs = r[1:mtry]
#     else
#         col_indcs = 1:p
#     end

#     for j in col_indcs
#         # Sorting used to be performed only when n <= 100, but doing it
#         # unconditionally improved fitting performance by 20%. It's a bit of a
#         # puzzle. Either it improved type-stability, or perhaps branch
#         # prediction is much better on a sorted sequence.
#         ord = sortperm(X[:, j])
#         x_j = X[ord, j]
#         y_ord = y[ord]

#         if n > 100
#             if VERSION >= v"0.4.0-dev"
#                 domain_j = quantile(x_j, linspace(0.01, 0.99, 99); sorted=true)
#             else  # sorted=true isn't supported on StatsBase's Julia 0.3 version
#                 domain_j = quantile(x_j, linspace(0.01, 0.99, 99))
#             end
#         else
#             domain_j = x_j
#         end
#         value, thresh = _best_mse_loss(y_ord, x_j, domain_j)

#         if value > best_val
#             best_val = value
#             best = (j, thresh)
#         end
#     end
#     return best
# end




function _split_mse_df{T<:Float64}(y::Vector{T}, X::DataFrame, mtry::Int)
    n, p = size(X)
    best = NO_BEST
    best_val = -Inf
    if mtry > 0
        r = randperm(p)
        col_indcs = r[1:mtry]
    else
        col_indcs = 1:p
    end
    for j in col_indcs
        keep_row = !isna(X[:, j])
        x_obs = convert(Vector, X[keep_row, j])
        if length(x_obs) == 0
            continue
        end

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



function find_na_cols(dat)
    cols_with_na = Array{Int, 1}(0)

    for j in 1:ncol(dat)
        if sum(isna(dat[:, j])) ≠ 0
            push!(cols_with_na, j)
        end
    end
    return cols_with_na
end


# function find_surrogates{T::BitArray}(left_node::T, y::Vector, X::Matrix, mtry::Int)



function build_stump{T <: Float64, U<:Real}(y::Vector{T}, X::Matrix{U})
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






function build_tree_df{T <: Float64}(y::Vector{T}, X::DataFrame, maxlabels = 5, mtry = 0, maxdepth = -1, max_surrogates = 5)
    n = nrow(X)
    if maxdepth < -1
        error("Unexpected value for maxdepth: $(maxdepth) (expected: maxdepth >= 0, or maxdepth = -1 for infinite depth)")
    end
    if length(y) <= maxlabels || maxdepth == 0          # stopping rules
        return Leaf(mean(y), y)
    end

    S = _split_mse_df(y, X, mtry)               # get [complete data] optimal split point

    if S == NO_BEST
        return Leaf(mean(y), y)
    end

    col_idx, thresh = S

    if max_surrogates ≠ 0
        cols_with_na = find_na_cols(X)                  # needed for all recursive steps since rows of X change
        surrogate_vars = Vector{Tuple{Int, Real}}(max_surrogates)
    end

    # if col_idx in cols_with_na
    if col_idx in 1:ncol(X)                             # find surrogates even when no missing data
        na_rows = isna(X[:, col_idx])
        split_with_na = Array{Any, 1}(n)                # vector of Bools with some NA values

        for i = 1:n
            split_with_na[i] = isna(X[i, col_idx]) ? NA : X[i, col_idx] < thresh
        end
        row_indcs = collect(1:n)[!na_rows]
        col_indcs = deleteat!(collect(1:ncol(X)), col_idx)

        # Here we need a function that splits so as to optimize agreement
        # with the `split_with_na` result for each observed `row_indcs`.
        surrogate_vars = surrogate_splits(split_with_na, X, row_indcs, col_indcs, 5)
        split = apply_surrogates(split_with_na, X, surrogate_vars)
    else
        split = X[:, col_idx] .< thresh
    end

    return Node(col_idx,
                thresh,
                surrogate_vars,
                build_tree_df(y[split], X[split,:], maxlabels, mtry, max(maxdepth-1, -1)),
                build_tree_df(y[!split], X[!split,:], maxlabels, mtry, max(maxdepth-1, -1)))
end


function build_forest_df{T <: Real}(y::Vector{T}, X::DataFrame, mtry, ntrees, maxlabels = 5, maxdepth = -1; nthreads = 1, oob_measure = "rsq")

    n = length(y)
    tree_arr = Array{Node, 1}(ntrees)

    if nthreads ≥ 2
        yhat_mat = fill(-Inf, (n, ntrees))       # use to compute oob-score
        
        @threads for t in 1:ntrees
            inds = sample(1:n, n)
            tree_arr[t] = build_tree_df(y[inds], X[inds, :], maxlabels, mtry, maxdepth)
            
            # get OOB indices to calculate OOB score
            oob_indcs = setdiff(collect(1:n), inds)
            yhat = apply_tree(tree_arr[t], X[oob_indcs, :])

            yhat_mat[oob_indcs, t] = yhat
        end
        oob_score = par_oob_score(yhat_mat, y, oob_measure)
    else
        yhat_mat = zeros(n, 2)                  # col 1 is numerator, col 2 is denominator
        yhat_mat[:, 1] = fill(-Inf, n)
        
        for t in 1:ntrees
            inds = sample(1:n, sample)
            tree_arr[t] = build_tree_df(y[inds], X[inds, :], maxlabels, mtry, maxdepth)

            # get OOB indices to calculate OOB score
            oob_indcs = setdiff(collect(1:n), inds)
            yhat = apply_tree(tree_arr[t], X[oob_indcs, :])
            
            if !all(isfinite(yhat_mat[oob_indcs, 1]))
                for idx in oob_indcs
                    if !isfinite(yhat_mat[idx, 1])
                        yhat_mat[oob_indcs, 1] = 0.0 
                    end 
                end 
            end            

            yhat_mat[oob_indcs, 1] += yhat 
            yhat_mat[oob_indcs, 2] += 1 
        end

        if all(isfinite(yhat_mat[:, 1]))
            oob_score = R2(y, yhat_mat[:, 1] ./ yhat_mat[:, 2])
        else 
            warn("Not enough trees built to compute OOB score. Some rows were never out-of-bag")
            return nothing
        end  
    end
    return (Ensemble(tree_arr), oob_score)
end


# This function is a helper used for getting oob-score for 
# parallel version of random forest regression. The parallel case 
# needs to be treated a bit differently to avoid threads overwriting 
# each other. NOTE: There might be a more elegant solution. 
function par_oob_score(yhat_mat, y, measure = "rsq")
    n, ntrees = size(yhat_mat)
    yhat = fill(-Inf, n)
    num_preds = zeros(Int, n)       # this will be our denomiator 
    
    for t = 1:ntrees
        for i = 1:n 
            if isfinite(yhat_mat[i, t])
                if !isfinite(yhat[i])
                    yhat[i] = 0.0
                end 
                yhat[i] += yhat_mat[i, t]
                num_preds[i] += 1
            end 
        end 
    end 
    if all(isfinite(yhat))
        yhat ./= num_preds 
    else 
        warn("Not enough trees built to compute OOB score. Some rows were never out-of-bag")
        return nothing
    end 
    if measure == "rsq"
        oob_score = R2(y, yhat) 
    elseif measure == "mse"
        oob_score = mean_squared_error(y, yhat) 
    end 
    return oob_score 
end 





