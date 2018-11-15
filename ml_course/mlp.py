"""
MLP
"""

from numpy import full

from sklearn.base import BaseEstimator, ClassifierMixin


class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100, ), momentum=0.0001, learning_rate=0.25):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.momentum = momentum
        self.learning_rate = learning_rate

    # Utilities

    def _gen_store_dimensions(self):
        i = 0

        # Input layer.
        prev_n_cnt = self.input_size_
        yield (
            self.input_size_,
            prev_n_cnt + 1,
        )
        i = i + 1

        # Hidden layers.
        self.begin_i_hidden_ = i

        for size in self.hidden_layer_sizes:
            yield (
                size,
                prev_n_cnt + 1,
            )
            i = i + 1
            prev_n_cnt = size

        # Output layer.
        self.begin_i_output_ = i
        yield (
            1,
            prev_n_cnt + 1,
        )

    def _gen_store(self, dimensions, value=0):
        return tuple(full(dim, value) for dim in dimensions)

    # Fit

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns a trained MLP model.
        """

        self.input_size_ = X.shape[1]

        dimens = tuple(dim for dim in self._gen_store_dimensions())
        dimens_n_only = tuple((dim[0], ) for dim in dimens)

        self.weight_ = self._gen_store(dimens)

        self.bias_ = self._gen_store(dimens_n_only, 1)
        self.y_ = self._gen_store(dimens_n_only)
        self.err_ = self._gen_store(dimens_n_only)

        return self
