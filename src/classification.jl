
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
            cur_split = x_obs .< thresh         # tons of mem-alloc here

            # ensure we have some observations in y
            if (length(y_obs[cur_split]) > 0) && (length(y_obs[!cur_split]) > 0)  
                value = _classifcation_error_loss(y_obs[cur_split], wgts_obs[cur_split]) + _classifcation_error_loss(y_obs[!cur_split], wgts_obs[!cur_split])
                if value > best_val
                    best_val = value
                    best = (j, thresh)
                end
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
    
    for j in column_indcs
        keep_row = !isna(X[:, j])               
        x_obs = convert(Vector, X[keep_row, j])
        y_obs = y[keep_row]

        if length(unique(x_obs)) ≥ 100
            x_obs = convert(Array{Float64, 1}, x_obs)          # can't be Array{Any,1} for quantile()
            domain_j::Array{Float64, 1} = quantile(x_obs, linspace(0.01, 0.99, 99))
        else
            domain_j = unique(x_obs)
        end
        if length(domain_j) > 1
            for thresh in domain_j[2:end]


                cur_split = x_obs .< thresh         # tons of mem-alloc here



                # ensure we have some observations in y
                if (any(cur_split)) && (!all(cur_split))       
                    value = _classifcation_error_loss(y_obs[cur_split]) + _classifcation_error_loss(y_obs[!cur_split])
                    if value > best_val
                        best_val = value
                        best = (j, thresh)
                    end
                end 
            end
        end 
    end
    return best
end


# n = 1000
# p = 20
# X = DataFrame(randn(n, p));
# y = rand([true, false], n);
# wgt = ones(Int, n);

# @code_warntype _split_classifcation_error_loss(y, X, collect(1:n), collect(1:p), wgt)

#
#
#
# @time surrogate_splits(y, X, collect(1:n), collect(1:p), 5, wgt)




#
