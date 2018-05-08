'''
Example of  wrapper for xgboost estimator
supports:
    *TODO: CatBoostClassifier
    *TODO: CatBoostRegressor

custom objective functions and weight not supported
'''


from .base import SHBaseEstimator
from .callback import  early_stop,EarlyStopException
#
#
import numpy as np

class SHCatBoostEstimator(SHBaseEstimator):
     def __init__(self,model):
         raise NotImplementedError

#         self.model = model
#         self.env = {'best_score':-np.infty,'best_iteration':-1,'earlier_stop':False}
#     def update(self,Xtrain,ytrain,Xval,yval,scoring,n_iterations):
#         pass