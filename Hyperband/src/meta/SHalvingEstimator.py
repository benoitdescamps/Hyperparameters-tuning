from abc import ABC,abstractmethod
import uuid
import pickle
import os

class SHalvingEstimator(ABC):
    def __init__(self):
        self.model = None
    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

    def save(self,name=None):
        if not(name):
            name = str(uuid.uuid4().hex)
        with open( os.path.join('cache',name+'.pickle'), "wb" ) as file:
            pickle.dump(self.model,file)
        return name

    def load(self,model_name):
        #add assert
        with open( os.path.join('cache',model_name+'.pickle'), "rb" ) as file:
            self.model = pickle.load(file)
    def remove(self,model_name):
        os.remove(os.path.join('cache',model_name+'.pickle') )
    def get_params(self):
        return self.model.get_params()

    def set_params(self,*args,**kwargs):
        self.model.set_params(*args,**kwargs)


    def n_iteration(self,ressource_name):
        return self.model.get_params()[ressource_name]

    @abstractmethod
    def update(self,X,y,n_iterations):
        return NotImplementedError


