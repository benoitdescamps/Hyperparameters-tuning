import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import pytest
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor

from ..SuccessiveHalving import SuccessiveHalving
from ..meta.xgb import SHXGBEstimator
from ..meta.sklearn import SHSklearnEstimator

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint, expon, lognorm


def test_xgb():
    '''
    Test whether the booster indeed gets updated
    :return:
    '''
    Xtrain = np.random.randn(100,10)
    ytrain = np.random.randint(0,2,100)

    Xval = np.random.randn(20, 10)
    yval = np.random.randint(0, 2, 20)


    init_n_estimators = 2
    n_new_iterations = 5

    classifier = SHXGBEstimator(model=XGBClassifier(n_estimators=4),\
                                      ressource_name='n_estimators')

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

def test_sklearn_():
    '''
    Test whether the booster indeed gets updated
    :return:
    '''
    Xtrain = np.random.randn(100,10)
    ytrain = np.random.randint(0,2,100)

    Xval = np.random.randn(20, 10)
    yval = np.random.randint(0, 2, 20)


    classifier = SHSklearnEstimator(model=RandomForestClassifier(n_estimators=4),\
                                      ressource_name='n_estimators')

    param_grid = {'max_depth': randint(1,10),
                   'min_impurity_decrease':lognorm(0.1)
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
