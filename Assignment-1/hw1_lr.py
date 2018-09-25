from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        feature_arr = numpy.append(numpy.ones((len(features),1)),numpy.array(features),axis=1)
        value_arr = numpy.array(values)
        feature_tr = feature_arr.transpose()
        self.w_lms = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(feature_tr,feature_arr)),feature_tr),value_arr)

    def predict(self, features: List[List[float]]) -> List[float]:
        return numpy.matmul(self.w_lms,numpy.append(numpy.ones((len(features),1)),numpy.array(features),axis=1).transpose())

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.w_lms


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        feature_arr = numpy.append(numpy.ones((len(features),1)),numpy.array(features),axis=1)
        value_arr = numpy.array(values)
        feature_tr = feature_arr.transpose()
        reg = self.alpha*numpy.identity(feature_arr.shape[1])
        addition = numpy.add(reg,numpy.matmul(feature_tr,feature_arr))
        self.w_lms_l2 = numpy.matmul(numpy.matmul(numpy.linalg.inv(addition),feature_tr),value_arr)

    def predict(self, features: List[List[float]]) -> List[float]:
        return numpy.matmul(self.w_lms_l2,numpy.append(numpy.ones((len(features),1)),numpy.array(features),axis=1).transpose())

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.w_lms_l2


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
