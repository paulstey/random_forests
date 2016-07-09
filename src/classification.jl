using Distributions

function _split_classifcation_error_loss(y::Vector, X::DataFrame, obs_row_indcs::Vector{Int}, column_indcs::Vector{Int}, weights::Vector)
    best = NO_BEST
    best_val = -Inf
    X = X[obs_row_indcs, :]
    y = y[obs_row_indcs]

    for j in column_indcs
        keep_row = !isna(X[:, j])                # use only non-NA values
        x_obs = convert(Vector, X[keep_row, j])
        y_obs = y[keep_row]
        wgts_obs = weights[keep_row]

        if length(unique(x_obs)) ≥ 100
            domain_j = quantile(x_obs, linspace(0.01, 0.99, 99))
        else
            domain_j = sort(unique(x_obs))
        end

        for thresh in domain_j[2:end]
            cur_split = x_obs .< thresh
            value = _classifcation_error_loss(y_obs[cur_split], wgts_obs[cur_split]) + _classifcation_error_loss(y_obs[!cur_split], wgts_obs[!cur_split])
            if value > best_val
                best_val = value
                best = (j, thresh)
            end
        end
    end
    return best
end



# This method is dispatched when weights are omitted. This
# allows us to compute the loss function 5x faster
function _split_classifcation_error_loss(y::Vector, X::DataFrame, obs_row_indcs::Vector{Int}, column_indcs::Vector{Int})
    best = NO_BEST
    best_val = -Inf
    X = X[obs_row_indcs, :]
    y = y[obs_row_indcs]

    # println(size(X))
    # println(column_indcs)

    for j in column_indcs
        keep_row = !isna(X[:, j])                # FIX THE BUG HERE!!!
        x_obs = convert(Vector, X[keep_row, j])
        y_obs = y[keep_row]

        if length(unique(x_obs)) ≥ 100
            domain_j = quantile(x_obs, linspace(0.01, 0.99, 99))
        else
            domain_j = sort(unique(x_obs))
        end

        for thresh in domain_j[2:end]
            cur_split = x_obs .< thresh
            value = _classifcation_error_loss(y_obs[cur_split]) + _classifcation_error_loss(y_obs[!cur_split])
            if value > best_val
                best_val = value
                best = (j, thresh)
            end
        end
    end
    return best
end

#
# n = 1000
# p = 20
# X = DataFrame(randn(n, p));
# y = rand([true, false], n);
# wgt = ones(Int, n);
#
# @time _split_classifcation_error_loss(y, X, collect(1:n), collect(1:p), wgt)
#
#
#
#
# @time surrogate_splits(y, X, collect(1:n), collect(1:p), 5, wgt)


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


#
