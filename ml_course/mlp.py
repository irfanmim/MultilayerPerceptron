"""
MLP
"""

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances


class MLP(BaseEstimator, ClusterMixin):

    def __init__(self,
                 k=2,
                 tolerance=0.0001,
                 max_iteration=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iteration = max_iteration

    def fit(self, data):
        self.centroids = {}
        self.label = []

        self.labels_ = self.label
        return self
