import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import pytest
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, StandardScaler
from .Transformers import DummyTransformer
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor

from ..meta.xgb import SHalvingXGBClassifier

@pytest.fixture(scope="module")
def dummyXGBClassifierPipeline():
    estimator = Pipeline([
        ('features', DummyTransformer()),
        ('classifier', XGBClassifier(n_estimators=2, max_depth=1))
    ])
    return estimator


def test_update_booster():
    '''
    Test whether the booster indeed gets updated
    :return:
    '''
    X = np.random.randn(100,10)
    y = np.random.randint(0,2,100)

    init_n_estimators = 2
    n_new_iterations = 5

    classifier = SHalvingXGBClassifier(n_estimators=init_n_estimators ,max_depth=1)
    classifier.fit(X,y)
    classifier.update(X,y,n_iterations=n_new_iterations)
    expected_n_estimators = init_n_estimators + n_new_iterations

    assert (classifier.get_params()['n_estimators']==expected_n_estimators)
