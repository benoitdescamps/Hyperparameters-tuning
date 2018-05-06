from .SHalvingEstimator import SHalvingEstimator


from xgboost import XGBClassifier,\
                    XGBRegressor, \
                    DMatrix

class SHalvingXGBClassifier(SHalvingEstimator):

    def __init__(self,*args,**kwargs):
        self.model = XGBClassifier(*args,**kwargs)

    def update(self,X,y,n_iterations):
        dtrain = DMatrix(data=X,label=y)
        for i in range(n_iterations):
            # note:
            # this is a get, but the internal booster in XGBClassifier is also updated
            # add unit test for controle if future updates
            self.model.get_booster().update(dtrain,iteration=self.model.n_estimators)
            self.model.n_estimators += 1

