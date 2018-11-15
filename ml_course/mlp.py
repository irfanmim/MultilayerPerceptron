"""
MLP
"""

import numpy as np

from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid_grad(y):
    return y * (1.0 - y)


class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100, ), momentum=0.0001, learning_rate=0.25):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.momentum = momentum
        self.learning_rate = learning_rate

    # Utilities

    def _gen_store_dimensions(self):
        i = 0

        # Input layer.
        self.i_input_ = i

        prev_n_cnt = self.input_size_
        yield (self.input_size_, prev_n_cnt,)
        i = i + 1

        # Hidden layers.
        self.i_hidden_begin_ = i

        for size in self.hidden_layer_sizes:
            yield (size, prev_n_cnt,)
            i = i + 1
            prev_n_cnt = size

        # Output layer.
        self.i_output_ = i
        yield (1, prev_n_cnt,)

    def _gen_store(self, dimensions, value=0.0):
        return tuple(np.full(dim, value) for dim in dimensions)

    # Fit

    def _feed_forward(self, X):
        data = X

        for i in range(len(self.y_)):
            sigma_values = np.sum(
                data * self.weight_[i], axis=1) + self.weight_bias_[i]

            data = self.y_[i]
            expit(sigma_values, out=data)

    def _local_gradient(self, X, y):
        # Output unit.
        weights = self.weight_[self.i_output_]
        err_terms = self.err_term_[self.i_output_]

        out_y = self.y_[self.i_output_][0]
        err_terms[0] = sigmoid_grad(out_y) * (y - out_y)

        # Hidden unit.
        for i in range(len(self.y_) - 2, -1, -1):
            d_y_arr = sigmoid_grad(self.y_[i])
            sig_arr = np.sum(
                np.expand_dims(err_terms, axis=1) * weights, axis=0)

            err_terms = self.err_term_[i]
            weights = self.weight_[i]

            np.multiply(d_y_arr, sig_arr, out=err_terms)

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
        self.weight_bias_ = self._gen_store(dimens_n_only)
        self.y_ = self._gen_store(dimens_n_only)
        self.err_ = self._gen_store(dimens_n_only)
        self.err_term_ = self._gen_store(dimens_n_only)

        return self
