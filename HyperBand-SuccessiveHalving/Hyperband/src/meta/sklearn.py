'''
Example of  wrapper for sklearn estimator
supports [any estimators with hyperparameter: warm-start]:
    *
    *

'''
import numpy as np
from .base import SHBaseEstimator

import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class SHSklearnEstimator(SHBaseEstimator):
    def __init__(self,model,ressource_name=None):
        self.model = model
        self.ressource_name = ressource_name
        self.env = None

    def update(self,Xtrain,ytrain,Xval,yval,scoring,n_iterations):
        self.set_params(**{'warm_start':True,self.ressource_name:n_iterations})
        self.model.fit(Xtrain,ytrain)
        logger.debug(self.ressource_name + ' : {}'.format(self.model.get_params()[self.ressource_name]) )
