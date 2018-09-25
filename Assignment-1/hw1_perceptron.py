from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        import math
        assert len(features) == len(labels)
        cnt = 0
        i = 0
        label = 0
        while cnt < self.max_iteration:
            mistakes = 0
            for i in range(len(features)):
                x = features[i]
                value = np.matmul(np.array(self.w).T,np.array(x))
                if value >= self.margin:
                    label = 1
                else: 
                    label = -1 
                if label != labels[i]:
                    mistakes = mistakes + 1
                    sum_of_sqares = 0.0
                    for j in x:
                        sum_of_sqares = sum_of_sqares + math.pow(j,2)
                    self.w = self.w + ((labels[i]/math.sqrt(sum_of_sqares))*np.array(x))
            if mistakes == 0:
                return True
            cnt = cnt + 1
        if cnt == self.max_iteration:
            return False
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        pred_labels = []
        for x in features:
            value = np.matmul(self.w,np.array(x).transpose())
            if value >= self.margin:
                pred_labels.append(1)
            else:
                pred_labels.append(-1)
        return pred_labels

    def get_weights(self) -> Tuple[List[float], float]:
        return self.w
    