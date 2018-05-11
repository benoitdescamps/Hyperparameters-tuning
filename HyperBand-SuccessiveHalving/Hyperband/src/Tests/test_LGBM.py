import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from ..meta.lgbm import SHLGBMEstimator
import pytest
import numpy as np
from sklearn.metrics import accuracy_score, make_scorer
from lightgbm import LGBMClassifier,LGBMRegressor,LGBMModel



def test_update_booster():
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

    scoring = make_scorer(accuracy_score)
    classifier = SHLGBMEstimator(model=LGBMModel(objective='binary',n_estimators=init_n_estimators ,max_depth=1),
                                 ressource_name='n_estimators')
    classifier.fit(Xtrain, ytrain)
    classifier.update(Xtrain,ytrain,Xval,yval,scoring=scoring,n_iterations=n_new_iterations)
    expected_n_estimators = init_n_estimators + n_new_iterations

    assert(classifier.get_params()['n_estimators']==expected_n_estimators)
