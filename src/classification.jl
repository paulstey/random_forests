

function _split_classifcation_error_loss(y::Vector, X::Matrix, weights::Vector)
    best = NO_BEST
    best_val = -Inf
    p = size(X, 2)

    for j in 1:p
        keep_row = !isna(X[:, j])
        x_obs = convert(Vector, X[keep_row, j])
        y_obs = y[keep_row]

        if length(unique(x_obs)) ≥ 100
            domain_j = quantile(x_obs, linspace(0.01, 0.99, 99))
        else
            domain_j = sort(unique(x_obs))
        end

        for thresh in domain_j[2:end]

            cur_split = x_obs .< thresh
            value = _classifcation_error_loss(y[cur_split], weights[cur_split]) + _classifcation_error_loss(y[!cur_split], weights[!cur_split])

            if value > best_val
                best_val = value
                best = (j, thresh)
            end
        end
    end
    return best
end

n = 1000
p = 20
X = rand(n, p);
y = rand(["yes", "no"], n);
wgt = repeat([1], inner = n);

@time _split_classifcation_error_loss(y, X, wgt)
