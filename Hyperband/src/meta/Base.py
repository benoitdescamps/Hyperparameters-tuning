from abc import ABC,abstractmethod
import uuid
import pickle
import os

from .core import EarlyStopException

class SHBaseEstimator(ABC):
    def __init__(self,model,ressource_name=None):
        self.model = None
        self.env = None
        self.ressource_name = ressource_name
    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

    def save(self,name=None):
        if not(name):
            name = str(uuid.uuid4().hex)
        with open( os.path.join('__cache__',name+'.pickle'), "wb" ) as file:
            pickle.dump(self,file)
        return name

    def load(self,model_name):
        #add assert
        with open( os.path.join('__cache__',model_name+'.pickle'), "rb" ) as file:
            tmp = pickle.load(file)
            self.model = tmp.model
            self.env = tmp.env
    def remove(self,model_name):
        os.remove(os.path.join('__cache__',model_name+'.pickle') )
    def get_params(self):
        return self.model.get_params()

    def set_params(self,*args,**kwargs):
        self.model.set_params(*args,**kwargs)


    def n_iteration(self,ressource_name):
        return self.model.get_params()[ressource_name]

    @abstractmethod
    def update(self,Xtrain,ytrain,Xval,yval,scoring,n_iterations):
        return NotImplementedError


