from abc import ABC,abstractmethod
import uuid
import pickle
import os

class SHalvingEstimator(ABC):

    def __init__(self):
        self.model = None
        self.name = str(uuid.uuid4().hex)
    def fit(self,X,y):
        self.model.fit(X,y)
        return self.model

    def predict(self,X):
        return self.model.predict(X)

    def save(self):
        with open( os.path.join('../cache',self.name+'.pickle', "wb" ) ) as file:
            pickle.dump(self.model,file)

    def load(self):
        with open( os.path.join('../cache',self.name+'.pickle', "wb" ) ) as file:
            self.model = pickle.load(file)

    def get_params(self):
        return self.model.get_params()

    @abstractmethod
    def update(self,X,y,n_iterations):
        return NotImplementedError

