from sklearn.base import BaseEstimator, TransformerMixin

class DummyTransformer(BaseEstimator, TransformerMixin):
    """Transformer which does absolutely nothing
    Attributes:
        None
    """
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X