# CSCI 567 Machine Learning

## Programming Assignment 5

### Dataset
In Problem 1, we will play with a small hidden Markov model. The parameters of the model
are given in hmm model.json.
In Problem 2, we will use a subset of MNIST dataset that you are already familiar with. As a reminder,
MNIST is a dataset of handwritten digits and each entry is a 28x28 grayscale image of a digit. We will
unroll those digits into one-dimensional arrays of size 784.

### Tasks
You will be asked to implement (1) hidden Markov models’ (HMM) inference procedures and
(2) Principal Component Analysis (PCA). Specifically, you will
- For Problem 1, finish implementing the function forward, backward, seqprob forward, seqprob backward, and viterbi. Refer to hmm.py for more information.
- Run the scripts hmm.sh, after you finish your implementation. hmm.sh will output hmm.txt.
- Add, commit, and push hmm.py and hmm.txt.
- For Problem 2, finish implementing the functions pca, decompress and reconstruction error. Refer to pca.py for more information.
- Run pca.py (i.e., do python3 pca.py), after you finish your implementation, it will generate pca output.txt.
- Add, commit, and push pca.py and pca output.txt.