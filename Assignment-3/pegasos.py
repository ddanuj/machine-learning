import json
import numpy as np


###### Q1.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm

    Return:
    - train_obj: the value of objective function in SVM primal formulation
    """
    # you need to fill in your solution here

    N, D = X.shape
    #First term - lambda / 2 + L2(w)^2
    first_term = (lamb / 2.0) * (np.linalg.norm(w, ord=2)**2)

    #Second Term - Sum(max(0,1-y*w.T*x))/N
    second_term = 0.0
    for i in range(N):
        second_term += max(0.0, (1.0 - np.matmul(X[i], y[i] * w)[0]))

    obj_value = first_term + (second_term / N)
    return obj_value


###### Q1.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the maximum number of iterations to update parameters

    Returns:
    - learnt w
    - traiin_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]

    train_obj = []

    ytrain = np.reshape(ytrain, (len(ytrain), 1))

    for iter in range(1, max_iterations + 1):
        A_t = np.floor(np.random.rand(k) * N).astype(
            int)  # index of the current mini-batch

        # you need to fill in your solution here
        # Set A+ - select indexes from A_t and select corresponding x and y values
        # Add only those to set A+ where y*w.T*x < 1
        X_t = []
        y_t = []
        for i in A_t:
            product = np.matmul(ytrain[i] * Xtrain[i], w)
            if (product < 1.0).all():
                X_t.append(Xtrain[i])
                y_t.append(ytrain[i])

        X_t = np.array(X_t)
        y_t = np.array(y_t)

        if X_t.shape[0] > 0:
            xy_t = np.zeros(X_t.shape[1])
            for i in range(X_t.shape[0]):
                xy_t += y_t[i] * X_t[i]
            xy_t = np.array(xy_t)
            xy_t = np.reshape(xy_t, (len(xy_t), 1))

            n_t = 1.0 / (lamb * iter)
            w = (1.0 - n_t * lamb) * w + (n_t / k) * xy_t
            lamb_sqrt_inv = 1 / np.sqrt(lamb)
            w_l2 = np.linalg.norm(w, ord=2)
            w = min(1.0, lamb_sqrt_inv / w_l2) * w
            
        train_obj.append(objective_function(Xtrain, ytrain, w, lamb))

    return w, train_obj


###### Q1.3 ######
def pegasos_test(Xtest, ytest, w, t=0.):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
    - t: threshold, when you get the prediction from SVM classifier, it should be real number from -1 to 1. Make all prediction less than t to -1 and otherwise make to 1 (Binarize)

    Returns:
    - test_acc: testing accuracy.
    """
    # you need to fill in your solution here
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)
    N, D = Xtest.shape

    ytest = np.reshape(ytest, (len(ytest), 1))

    correct_sample = 0
    thic_class = -1

    for i in range(N):
        if np.matmul(Xtest[i], w)[0] < t:
            thic_class = -1
        else:
            thic_class = 1

        if thic_class == ytest[i]:
            correct_sample += 1

    test_acc = correct_sample / N
    return test_acc


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""


def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
        data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set[
        'valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(
        dataset='mnist_subset.json')

    max_iterations = 500
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(
            Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(
            Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(
            Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(
            Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist()  # results on mnist
    print('mnist test acc \n')
    for key, value in test_acc.items():
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
