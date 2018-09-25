# CSCI 567 Machine Learning

## Programming Assignment 3

### Dataset
We will use mnist subset (images of handwritten digits from 0 to 9). This is the same subset
of the full MNIST that we used for Homework 1 and Homework 2. As before, the dataset is stored in a
JSON-formated file mnist subset.json. You can access its training, validation, and test splits using the keys
‘train’, ‘valid’, and ‘test’, respectively. For example, suppose we load mnist subset.json to the variable x.
Then, x[0train0] refers to the training set of mnist subset. This set is a list with two elements: x[0train0][0]
containing the features of size N (samples) D (dimension of features), and x[0train0][1] containing the
corresponding labels of size N.

### Tasks
You will be asked to implement the linear support vector machine (SVM) for binary classification
(Sect. 1). Specifically, you will
- finish implementing the following three functions—objective function, pegasos train, and pegasos test—in pegasos.py. Refer to pegasos.py and Sect. 1 for more information.
- Run the script pegasos.sh after you finish your implementation. This will output pegasos.json.
- add, commit, and push (1) the pegasos.py, and (2) the pegasos.json file that you have created. 

You will be asked to implement the boosting algorithms with decision stump (Sect. 2). Specifically, you will
- finish implementing the following classes — boosting, decision stump, AdaBoost and Logit Boost. Refer to boosting.py, decision stump.py and Sect. 2 for more information.
- Run the script boosting.sh after you finish your implementation. This will output boosting.json.
- Add, commit, and push (1) the boosting.py, (2) the decision stump.py, and (3) the boosting.json.

You will be asked to implement the decision tree classifier (Sect. 3). Specifically, you will
- finish implementing the following classes—DecisionTree, TreeNode. Refer to decision tree.py and Sect. 3 for more information.
- Run the script decision tree.sh after you finish your implementation. This will output decision tree.json.
- Add, commit, and push (1) the decision tree.py, (2) the decision tree.json