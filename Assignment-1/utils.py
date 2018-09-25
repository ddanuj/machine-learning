from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    se_list=list()
    for i in range(n):
        se_list.append(np.power(y_true[i]-y_pred[i],2.0))
    sum_list = sum(se_list)
    return sum_list / n


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)

    positive_class = 1
    negative_class = 0

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(predicted_labels)):
        if predicted_labels[i] == real_labels[i]:
            if predicted_labels[i] == positive_class:
                true_positives = true_positives + 1
        else:
            if predicted_labels[i] == negative_class:
                false_negatives = false_negatives + 1
            else:
                false_positives = false_positives + 1
    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0.0
    
    return f1


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    new_feature_arr = np.zeros((len(features),1))
    cnt = 1
    while (cnt <= k):
        new_features = []
        for x in features:
            new_features.append([float("{0:.6f}".format(pow(x[0],cnt)))])
        new_feature_arr = np.append(new_feature_arr,np.array(new_features),axis=1)
        cnt = cnt + 1
    new_feature_arr = np.delete(new_feature_arr,0,1)
    return new_feature_arr.tolist()


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    import math
    assert len(point1) == len(point2)
    sum_of_squares = 0.0
    for i in range(len(point1)):
        sum_of_squares = sum_of_squares + pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_of_squares)


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    assert len(point1) == len(point2)
    sum_of_dots = 0.0
    for i in range(len(point1)):
        sum_of_dots = sum_of_dots + (point1[i]*point2[i])
    return sum_of_dots


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    import math
    assert len(point1) == len(point2)
    ed = euclidean_distance(point1,point2)
    gd = -math.exp(-pow(ed,2)/2.0)
    return gd


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        import math
        norm_features = np.array(np.ones((1,len(features[0]))))
        for x in features:
            sum_of_squares = 0.0
            if np.count_nonzero(x) > 0:
                for i in range(len(x)):
                    sum_of_squares = sum_of_squares + pow(x[i], 2.0)
                norm_features = np.append(norm_features,[np.divide(np.array(x),math.sqrt(sum_of_squares))],axis=0)
            else:
                np.append(norm_features,[np.array(x)],axis=0)
        norm_features = np.delete(norm_features,0,0)
        return norm_features


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first_call = True
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        min_max_norm_features = np.array(np.ones((1,len(features[0]))))
        if self.first_call:
            self.scaler_vector = np.zeros(len(features[0]))
            for feature in features:
                for component in feature:
                    if component > self.scaler_vector[feature.index(component)]:
                        self.scaler_vector[feature.index(component)] = component
            self.first_call = False
        for feature in features:
            min_max_norm_features = np.append(min_max_norm_features,[np.divide(np.array(feature),np.array(self.scaler_vector))],axis=0)
        min_max_norm_features = np.delete(min_max_norm_features,0,0)
        return min_max_norm_features
