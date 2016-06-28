

function _split_classifcation_error_loss(y::Vector, X::Matrix, weights::Vector)
    best = NO_BEST
    best_val = -Inf
    p = size(X, 2)

    for j in 1:p
        domain_j = sort(unique(X[:, j]))

        for thresh in domain_j[2:end]
            cur_split = X[:, j] .< thresh
            value = _classifcation_error_loss(y[cur_split], weights[cur_split]) + _classifcation_error_loss(y[!cur_split], weights[!cur_split])
            if value > best_val
                best_val = value
                best = (i, thresh)
            end
        end
    end
    return best
end
