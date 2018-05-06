from .SHalvingEstimator import SHalvingEstimator

from xgboost import XGBClassifier,\
                    XGBRegressor, \
                    DMatrix

class SHalvingXGBClassifier(SHalvingEstimator):

    def __init__(self,*args,**kwargs):
        self.model = XGBClassifier(*args,**kwargs)

    def update(self,X,y,n_iterations):
        dtrain = DMatrix(data=X,label=y)
        current_iteration = self.model.n_estimators
        for i in n_iterations:
            self.model.get_booster().up
            self.model.n_estimators += 1


