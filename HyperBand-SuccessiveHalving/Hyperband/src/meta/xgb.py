'''
Example of  wrapper for xgboost estimator
supports:
    *XGBClassifier
    *XGBRegressor

custom objective functions and weight not supported
'''
from .base import SHBaseEstimator
from .callback import  early_stop,EarlyStopException

from xgboost import DMatrix

import numpy as np

class SHXGBEstimator(SHBaseEstimator):
    def __init__(self,model,ressource_name='n_estimators'):
        self.model = model
        self.env = {'best_score':-np.infty,'best_iteration':-1,'earlier_stop':False}
        self.ressource_name = ressource_name
    def update(self,Xtrain,ytrain,Xval,yval,scoring,n_iterations):
        dtrain = DMatrix(data=Xtrain,label=ytrain)

        early_stop_callback = early_stop()

        if not(self.env['earlier_stop']):
            for i in range(n_iterations-self.model.n_estimators):
                # note:
                # this is a get, but the internal booster in XGBClassifier is also updated
                # add unit test for controle if future updates
                self.model.get_booster().update(dtrain,iteration=self.model.n_estimators)
                self.model.n_estimators += 1

                score = scoring(self,Xval,yval)

                if score >  self.env['best_score']:
                    self.env['best_score'] = score
                    self.env['best_iteration'] = self.model.n_estimators
                try:
                    early_stop_callback(env=self.env,
                                        score=score,
                                        iteration=self.model.n_estimators)
                except EarlyStopException:
                    print('Update Stopped Earlier! @ {} instead of {}'.format(self.model.n_estimators,n_iterations) )
                    self.env['earlier_stop'] = True
                    break






