from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function
        self.features = []
        self.labels = []

    def train(self, features: List[List[float]], labels: List[int]):
        self.features = features
        self.labels = labels

    def predict(self, features: List[List[float]]) -> List[int]:
        import operator
        from statistics import mode, StatisticsError
        distance = {}
        pred_labels = []
        for x_test in features:
            cnt = 0
            for x_train in self.features:
                distance[cnt] = self.distance_function(x_test, x_train)
                cnt = cnt + 1
            top_k = dict(sorted(distance.items(), key=operator.itemgetter(1))[:self.k]).keys()
            label_set = []
            for k in top_k:
                label_set.append(self.labels[k])
            try:
                pred_labels.append(mode(label_set))
            except StatisticsError:
                pred_labels.append(1)
        return pred_labels


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
