include("EnsembleMethods.jl")

dset = readtable("../data/concrete_data.csv")

X = dset[:, 1:8]
y = convert(Array, dset[:, 9])

@time fm1 = build_forest_df(y, X, 3, 5; nthreads = 1, n_surrogates = 0);
Profile.clear_malloc_data() 
@time fm1 = build_forest_df(y, X, 3, 50; nthreads = 1, n_surrogates = 0);




using DecisionTree

X = convert(Array, dset[:, 1:8])
y = convert(Array, dset[:, 9])
@time fm2 = build_forest(y, X, 3, 50);


