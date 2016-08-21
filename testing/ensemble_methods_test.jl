include("EnsembleMethods.jl")

d_raw = readtable("../data/concrete_data.csv")
dset = as_floats(d_raw)

X = dset[:, 1:8]
y = convert(Array, dset[:, 9])

@time fm1 = build_forest_df(y, X, 3, 5; nthreads = 1, n_surrogates = 0);
Profile.clear_malloc_data() 
@time fm1 = build_forest_df(y, X, 3, 50; nthreads = 1, n_surrogates = 0);



n = 1000
p = 20
X = DataFrame(randn(n, p));
y = rand([true, false], n);
wgt = ones(Int, n);

@code_warntype _split_classifcation_error_loss(y, X, collect(1:n), collect(1:p), wgt)



using DecisionTree

X = convert(Array, dset[:, 1:8])
y = convert(Array, dset[:, 9])
@time fm2 = build_forest(y, X, 3, 50);


