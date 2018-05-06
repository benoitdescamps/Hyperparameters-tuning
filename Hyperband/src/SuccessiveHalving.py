'''
Based on the paper Non-stochastic best arm identiﬁcation and hyperparameter optimization.
by K. Jamieson and A. Talwalkar.
'''
from sklearn.model_selection import ParameterSampler
import numpy as np
import math
class SuccessiveHalving(object):
    """Applies successhalving on a model for n configurations max r ressources.

    Args:
        n: integer:
            number of  hyperparameter configurations to explore

        r: integer:
            maximum number of ressources.

        param_grid: dict:
            Dictionary where the keys are parameters and values are distributions from which a parameter is to be sampled. Distributions either have to provide a ``rvs`` function to sample from them, or can be given as a list of values, where a uniform distribution is assumed. e.g.: This could be a multiple of boosting iterations
            must be of the form:
            {
            'param_1': distribution_n,
            etc...
            'param_n': distribution_n,
            '__ressource__':'name_ressource_parameter'
            }
        seed: integer

        ressource_name: str
            Name of the ressource parameter
            e.g. for XGBClassifier this is 'n_estimators"

    """
    def __init__(self,estimator,n,r,param_grid,
                 ressource_name = 'n_estimators',
                 ressource_unit = 10,
                 scoring=None, n_jobs=1,cv=None,seed=0):
        self.estimator = estimator
        self.n = n
        self.r = r
        self.param_grid = param_grid
        self.ressource_name = ressource_name
        self.ressource_unit = ressource_unit
        self.seed = seed
        self.scoring = scoring
        self.n_jobs = n_jobs

        self.history = list()

    def apply(self,
              Xtrain,ytrain,Xval,yval
              ):
        """Apply Successive Halving:
             1. evaluate the performance of all conﬁgurations
             2. throw out the worst half
             3. return to 1. until one conﬁgurations remains.
            Args:
                X (numpy array): data
                y (numpy array): target
            Returns:
                    best configuration

            """
        T = self._get_hyperparameter_configurations(self.n)
        current_model_names= [None for i  in range(len(T))]
        first_fit =True

        n_iterations = self.r
        while (len(T) > 1):
            L = list()
            new_history_entry = list()
            for i in range(len(T)):
                model_name,  score = self._run_then_return_val_loss(ri=n_iterations,
                                               Xtrain=Xtrain, ytrain=ytrain,
                                               Xval=Xval, yval=yval,
                                               first_fit=first_fit,
                                               model_name=current_model_names[i],
                                               **T[i])
                if first_fit:
                    current_model_names[i] = model_name
                L.append(score)
                new_history_entry.append({
                    'model_name': model_name,
                 'score': score,
                 **T[i]})

            self.history.append(new_history_entry)
            if first_fit:
                first_fit= False

            remaining_model_names,T = self._get_top_k(T,current_model_names, L, k=math.ceil(len(T) / 2))

            self._clean_cache(list(set(current_model_names)-set(remaining_model_names)))
            current_model_names = remaining_model_names

            n_iterations+=self.ressource_unit
        return T

    def _get_top_k(self,T,list_model_names,L,k):
        indices =  np.argsort(L)[::-1][:k:]
        return [list_model_names[i] for i in indices],[T[i] for i in indices] #highest score

    def _clean_cache(self,list_model_names):
        for name in list_model_names:
            self.estimator.remove(name)

    def _get_hyperparameter_configurations(self,n):
        """
            Args:
                None
            Returns:
                    n randomly sampled hyperparameter configutations

            """
        np.random.seed(self.seed)
        return list(ParameterSampler(self.param_grid, n_iter=n))

    def _run_then_return_val_loss(self,
                                  ri,
                                  Xtrain,ytrain,
                                  Xval,yval,
                                  model_name = None,
                                  first_fit=False,
                                  **params):

        """
            set the parameters of the model and return the score
            the models are of the type SHalvingEstimators
            Args:
                None
            Returns:
                    n randomly sampled hyperparameter configutations

        """
        if first_fit:
            self.estimator.set_params(**params,**{self.ressource_name:ri})
            self.estimator.fit(Xtrain,ytrain)
            model_name = self.estimator.save()

        elif model_name:
            self.estimator.load(model_name)

            assert(ri>self.estimator.n_iteration(self.ressource_name)), 'The new ressource value ri should be greater than the current ressource value {}:{} of the estimator )'.format(self.ressource_name,self.estimator.n_iteration(self.ressource_name) )

            self.estimator.update(Xtrain,ytrain,n_iterations=ri-self.estimator.n_iteration(ressource_name=self.ressource_name)  )
            self.estimator.save(name=model_name)

        else:
            raise NotImplementedError

        return model_name,self.scoring(self.estimator,Xval,yval)





