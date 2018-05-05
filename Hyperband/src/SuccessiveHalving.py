'''
Based on the paper Non-stochastic best arm identiﬁcation and hyperparameter optimization.
by K. Jamieson and A. Talwalkar.
'''
from sklearn.model_selection import ParameterSampler
import numpy as np
class SuccessiveHalving(object):
    """Applies successhalving on a model for n configurations max r ressources.

    Args:
        n: integer:
            number of  hyperparameter configurations to explore

        r: integer:
            maximum number of ressources.

        param_grid: dict:
            Dictionary where the keys are parameters and values are distributions from which a parameter is to be sampled. Distributions either have to provide a ``rvs`` function to sample from them, or can be given as a list of values, where a uniform distribution is assumed. e.g.: This could be a multiple of boosting iterations

        seed: integer

    """
    def __init__(self,n,r,param_grid,scoring=None, n_jobs=1,cv=None,seed=0):
        self.n = n
        self.r = r
        self.param_grid = param_grid
        self.seed = seed
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
    def apply(self,
              estimator,
              X,
              y):
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
        T = self._get_hyperparameter_configurations(self,self.n)

        raise NotImplementedError

    def _get_top_k(self,T,L,k):
        return [T[i] for i in np.argsort(L)[::-1][:k:]] #highest score

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
                                  estimator,
                                  ri,
                                  **params):

        """
            set the parameters of the model and return the score
            Args:
                None
            Returns:
                    n randomly sampled hyperparameter configutations

        """
        raise NotImplementedError


