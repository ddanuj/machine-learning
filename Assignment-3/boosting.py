import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod


class Boosting(Classifier):
    # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T

        self.clfs_picked = []  # list of classifiers h_t for t=0,...,T-1
        self.betas = []  # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:
        ########################################################
        # TODO: implement "predict"
        ########################################################
        features = np.array(features)
        N, D = features.shape
        sum_beta_clf = np.zeros((N, 1))
        sign_arr = []
        for i in range(len(self.clfs_picked)):
            clfs_picked_predict = np.array(
                self.clfs_picked[i].predict(features))
            sum_beta_clf += self.betas[i] * np.reshape(clfs_picked_predict, (
                len(clfs_picked_predict), 1))
        for i in sum_beta_clf:
            if i > 0:
                sign_arr.append(1)
            else:
                sign_arr.append(-1)
        return sign_arr


class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return

    def sign_complement(self, n1: int, n2: int) -> int:
        if n2 != n1:
            return 1
        else:
            return 0

    def train(self, features: List[List[float]], labels: List[int]):
        ############################################################
        # TODO: implement "train"
        ############################################################
        features = np.array(features)
        N, D = features.shape
        w = np.full((N, 1), 1.0 / N, dtype=float)
        for iter in range(0, self.T - 1):
            entropy_picked = 10000.0
            for h in self.clfs:
                h_predict = h.predict(features)
                entropy = 0.0
                for i in range(N):
                    entropy += w[i] * self.sign_complement(
                        labels[i], h_predict[i])
                if entropy < entropy_picked:
                    entropy_picked = entropy
                    clfs_picked = h
            self.clfs_picked.append(clfs_picked)
            beta = 0.5 * np.log((1.0 - entropy_picked) / entropy_picked)
            self.betas.append(beta)
            clfs_picked_predict = clfs_picked.predict(features)
            for i in range(N):
                if labels[i] == clfs_picked_predict[i]:
                    w[i] = w[i] * np.exp(-beta)
                else:
                    w[i] = w[i] * np.exp(beta)
            w = np.divide(w, np.sum(w))

    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)


class LogitBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "LogitBoost"
        return

    def train(self, features: List[List[float]], labels: List[int]):
        ############################################################
        # TODO: implement "train"
        ############################################################
        features = np.array(features)
        N, D = features.shape
        pi = np.full((N, 1), 0.5, dtype=float)
        z = np.zeros((N, 1))
        w = np.zeros((N, 1))
        f = np.zeros((N, 1))
        for iter in range(0, self.T - 1):
            for i in range(N):
                z[i] = (((labels[i] + 1) / 2.0) - pi[i]) / (pi[i] *
                                                            (1.0 - pi[i]))
                w[i] = pi[i] * (1.0 - pi[i])
            entropy_picked = 10000.0
            for h in self.clfs:
                h_predict = h.predict(features)
                entropy = 0.0
                for i in range(N):
                    entropy += w[i] * (z[i] - h_predict[i])**2
                if entropy < entropy_picked:
                    entropy_picked = entropy
                    clfs_picked = h
            self.clfs_picked.append(clfs_picked)
            self.betas.append(0.5)
            f += 0.5 * np.reshape(np.array(h_predict), (len(h_predict), 1))
            for i in range(N):
                pi[i] = 1.0 / (1.0 + np.exp(-2.0 * f[i]))

    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
