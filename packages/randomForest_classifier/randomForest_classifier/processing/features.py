import numpy as np
from itertools import compress
from sklearn.base import BaseEstimator, TransformerMixin

from randomForest_classifier.processing.errors import InvalidModelInputError


class LogTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transformer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()
        print(X.shape)
        # check that the values are non-negative for log transform
        if not (X[self.variables] > 0).all().all():
            #vars_ = self.variables[(X[self.variables] <= 0).any()]
            vars_ = list(compress(self.variables, (X[self.variables] <= 0).any().tolist()))
            raise InvalidModelInputError(
                f"Variables contain zero or negative values, "
                f"can't apply log for vars: {vars_}"
            )

        for feature in self.variables:
            X[feature] = np.round(np.log(X[feature]),2)
        return X
