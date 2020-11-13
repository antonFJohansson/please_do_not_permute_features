


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern

from sklearn.datasets import load_boston
import random
import numpy as np
from featureImportance import feature_importance

X = load_boston()
y_data = X.target
feature_names = X.feature_names
X = X.data
m = np.mean(X, axis = 0)
st = np.std(X, axis = 0)
X = (X - m) / st

init_l = tuple(13*[1])
kernel =  Matern(length_scale=init_l)

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha = 1e-3).fit(X, y_data)

fi = feature_importance(gpr, X, y_data, feature_names)
fi.get_PD_plot(feature_ind = [0,-1])
fi.get_ICE_plot(feature_ind = [0,-1], num_samples = 10)
fi.get_VI_plot()
