

function _split_classifcation_error_loss(y::Vector, X::Matrix, weights::Vector)
    best = NO_BEST
    best_val = -Inf
    n, p = size(X)

    for j in 1:p
        x_j = X[:, j]

        if length(unique(x_j)) ≥ 100
            domain_j = quantile(x_j, linspace(0.01, 0.99, 99))
        else
            domain_j = sort(unique(x_j))
        end

        for thresh in domain_j[2:end]

            cur_split = x_j .< thresh
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
