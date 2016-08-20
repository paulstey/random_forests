import pandas as pd 
from sklearn.ensemble.forest import RandomForestRegressor
import time

dset = pd.read_csv("./data/concrete_data.csv")
X = dset.iloc[:, 0:7]
y = dset.iloc[:, 8]


estimator = RandomForestRegressor(max_features = 3, n_estimators = 50, n_jobs = 1, oob_score = True)

t0 = time.time()
estimator.fit(X, y)
print(time.time() - t0)

