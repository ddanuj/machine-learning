# CSCI 567 Machine Learning

## Programming Assignment 2

### Dataset

We will use mnist subset (images of handwritten digits from 0 to 9). The dataset is stored in
a JSON-formated file mnist subset.json. You can access its training, validation, and test splits using
the keys `train', `valid', and `test', respectively. For example, suppose we load mnist subset.json
to the variable x. Then, x[0train0] refers to the training set of mnist subset. This set is a list with
two elements: x[0train0][0] containing the features of size N (samples) D (dimension of features),
and x[0train0][1] containing the corresponding labels of size N.

### Tasks

You will be asked to implement binary and multiclass classification and neural networks. Specifically, you will
- Finish the implementation of all python functions in our template codes.
- Run your code by calling the specified scripts to generate output files.
- Add, commit, and push (1) all *.py files, and (2) all *.json and *.out files that you have amended or created.