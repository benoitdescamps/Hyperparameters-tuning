import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import pytest
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, StandardScaler

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor

from SuccessiveHalving import SuccessiveHalving
from meta.xgb import SHalvingXGBEstimator

from sklearn.metrics import accuracy_score, make_scorer

from scipy.stats import randint, expon, lognorm


Xtrain = np.random.randn(100,10)
ytrain = np.random.randint(0,2,100)

Xval = np.random.randn(20, 10)
yval = np.random.randint(0, 2, 20)


init_n_estimators = 2
n_new_iterations = 5

classifier = SHalvingXGBEstimator(model=XGBClassifier(n_estimators=4))

param_grid = {'max_depth': randint(1,10),
                'learning_rate':lognorm(0.1)
              }
scoring = make_scorer(accuracy_score)
successiveHalving = SuccessiveHalving(
    estimator=classifier,
    n = 10,
    r = 100,
    param_grid=param_grid,
    ressource_name='n_estimators',
    scoring=scoring,
    n_jobs=1,
    cv=None,
    seed=0
)

T = successiveHalving.apply(Xtrain,ytrain,Xval,yval)
print(T)

assert(True)
