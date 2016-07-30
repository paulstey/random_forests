using RDatasets

include("EnsembleMethods.jl")



n = 1000
p = 20
X = DataFrame(rand(n, p));
y = rand([true, false], n);
wgt = repeat([1], inner = n);

@time _split_classifcation_error_loss(y, X, wgt, collect(1:p))
@profile _split_classifcation_error_loss(y, X, wgt, collect(1:p))


include("EnsembleMethods.jl")


# testing build_tree_df()
n = 30
p = 5
X = DataFrame(randn(n, p));
y = randn(n);
X_mis = add_missing(X, 0.10);
build_tree_df(y, X_mis)

# test build_tree_df() and apply_tree()
n = 20
p = 10
X = DataFrame(randn(n, p));
β = [0.002, 0.001, 0.003, 0.001, 0.001, 0.02, 0.01, 0.03, 5.0, 123.5]
ε = randn(n)
y = ones(n) .+ Array(X) * β .+ ε
X_mis = add_missing(X, 0.95)


fm1 = build_tree_df(y, X_mis)
res1 = apply_tree(fm1, X_mis)
R2(y, res1)
mean_squared_error(y, res1)

# testing build_forest_df()
fm2 = build_forest_df(y, X_mis, p, 150; nthreads = 1)
# res = apply_forest(fm2, X)





# ds = dataset("datasets", "airquality")
# ds1 = ds[:, 2:6]
# y = convert(Vector, mean_impute(ds[:, 1]))

# fm1 = build_tree_df(y, ds1)

# fm3 = build_forest_df(y, ds1, 5, 50; nthreads = 2)

# y_hat2 = apply_forest(fm3, convert(Matrix{Any}, ds1))
