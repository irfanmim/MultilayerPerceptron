"""
MLP
"""

import numpy as np

from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid_grad(y):
    return y * (1.0 - y)


class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100, ), momentum=0.0001, learning_rate=0.25, max_iteration=10000):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration

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
        buf_a = self.weight_current_i_

        for i in range(len(self.y_)):
            sigma_values = np.sum(
                data * self.weight_[buf_a][i], axis=1) + self.weight_bias_[buf_a][i]

            data = self.y_[i]
            expit(sigma_values, out=data)

    def _local_gradient(self, X, y):
        buf_a = self.weight_current_i_

        # Output unit.
        weights = self.weight_[buf_a][self.i_output_]
        err_terms = self.err_term_[self.i_output_]

        out_y = self.y_[self.i_output_][0]
        err_terms[0] = sigmoid_grad(out_y) * (y - out_y)

        # Hidden unit.
        for i in range(len(self.y_) - 2, -1, -1):
            d_y_arr = sigmoid_grad(self.y_[i])
            sig_arr = np.sum(
                np.expand_dims(err_terms, axis=1) * weights, axis=0)

            err_terms = self.err_term_[i]
            weights = self.weight_[buf_a][i]

            np.multiply(d_y_arr, sig_arr, out=err_terms)

    def _update_weights(self, X):
        data = X

        # Swap buffer index
        buf_a = self.weight_current_i_
        buf_b = (buf_a + 1) % 2

        self.weight_current_i_ = buf_b

        for i in range(len(self.y_)):
            # Data.
            momentum = self.weight_[buf_b][i] * self.momentum
            learning = self.learning_rate * np.expand_dims(self.err_term_[i], axis=1) * np.expand_dims(data, axis=0)

            weight_ref = self.weight_[buf_a][i]
            weight_ref += momentum + learning

            # Bias.
            momentum = self.weight_bias_[buf_b][i] * self.momentum
            learning = self.learning_rate * self.err_term_[i]

            weight_ref = self.weight_bias_[buf_a][i]
            weight_ref += momentum + learning

            data = self.y_[i]

    def fit(self, X, y):
        self.input_size_ = X.shape[1]

        dimens = tuple(dim for dim in self._gen_store_dimensions())
        dimens_n_only = tuple((dim[0], ) for dim in dimens)

        self.weight_current_i_ = 0

        self.weight_ = (self._gen_store(dimens), self._gen_store(dimens),)
        self.weight_bias_ = (
            self._gen_store(dimens_n_only), self._gen_store(dimens_n_only))

        self.y_ = self._gen_store(dimens_n_only)
        self.err_ = self._gen_store(dimens_n_only)
        self.err_term_ = self._gen_store(dimens_n_only)

        for _ in range(self.max_iteration):
            err = 0
            for i in range(X.shape[0]):
                self._feed_forward(X[i])
                self._local_gradient(X[i], y[i])
                self._update_weights(X[i])

                err += (y[i] - self.y_[self.i_output_][0]) ** 2

            self.err_value_ = err * 0.5

        return self

    def predict(self, X, y=None):
        result = np.empty(X.shape[0])

        for i in range(X.shape[0]):
            self._feed_forward(X[i])
            result[i] = self.y_[self.i_output_][0]

        return result
