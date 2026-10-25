"""How well does a Gaussian-process surrogate predict post-route LUTs (and FFs) of designs it has not seen?
10-fold cross-validation over every design built so far (out/dse_results.csv, validation split), against a
mean-value and a linear-regression baseline. Usage: python surrogate_check.py"""
import os

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, 'out', 'dse_results.csv'))
d = d[d.split == 'val']
X = d[['H1', 'H2', 'WW', 'TAU']].to_numpy(float)
cv = KFold(10, shuffle=True, random_state=0)
gp = make_pipeline(StandardScaler(), GaussianProcessRegressor(
    ConstantKernel() * Matern(length_scale=np.ones(4), nu=2.5) + WhiteKernel(), normalize_y=True, random_state=0))
print(f'{len(d)} designs, LUTs {d.LUTs.min():.0f} to {d.LUTs.max():.0f}')
for target in ('LUTs', 'FFs'):
    y = d[target].to_numpy(float)
    for name, model in (('mean', None), ('linear', LinearRegression()), ('GP', gp)):
        p = np.full_like(y, y.mean()) if model is None else cross_val_predict(model, X, y, cv=cv)
        err = np.abs(p - y)
        r2 = 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f'{target:4s} {name:6s} MAE {err.mean():6.1f}  ({100 * (err / y).mean():4.1f}%)  '
              f'within 5%: {100 * (err / y <= 0.05).mean():5.1f}%  R2 {r2:5.2f}')
