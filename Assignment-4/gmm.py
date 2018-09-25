import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            k_means = KMeans(self.n_cluster, self.max_iter, self.e)
            centroids, membership, cnt = k_means.fit(x)
            self.means = centroids
            r = np.eye(len(centroids))[membership]
            variances = []
            pi = []
            for k in range(len(centroids)):
                sub_k = np.subtract(x, centroids[k])
                sum_k = np.sum(r[:, k])
                r_k = r[:, k].reshape((N, 1))
                var_n_t1 = np.multiply(r_k, sub_k)
                var_n_t2 = np.matmul(var_n_t1.T, sub_k)
                variances.append(var_n_t2 / sum_k)
                pi.append(sum_k / N)
            self.variances = np.array(variances)
            self.pi_k = np.array(pi)
            '''raise Exception(
                'Implement initialization of variances, means, pi_k using k-means')'''
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            random_indices = np.random.choice(N, size=self.n_cluster)
            self.means = x[random_indices]
            variances = []
            pi = []
            for k in range(len(self.means)):
                variances.append(np.identity(D))
                pi.append(1.0 / len(self.means))
            self.variances = np.array(variances)
            self.pi_k = np.array(pi)
            '''raise Exception(
                'Implement initialization of variances, means, pi_k randomly')'''
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        #Log Likelihood
        ll_x_old = self.compute_log_likelihood(x)
        cnt = 0
        while cnt < self.max_iter:
            #E step
            r = np.empty((N, self.n_cluster))
            for i in range(N):
                # Denominator - Sum i,k
                sum_k = 0.0
                for k in range(len(self.means)):
                    coef_k = (1.0 / np.sqrt(
                        np.power(2 * np.pi, D) * np.linalg.det(
                            self.variances[k])))
                    sub_k = np.subtract(x[i], self.means[k])
                    exp_t1 = np.matmul(-0.5 * sub_k,
                                       np.linalg.inv(self.variances[k]))
                    exp_t2 = np.matmul(exp_t1, sub_k.T)
                    sum_k += coef_k * np.exp(exp_t2)
                # Numerator - i,k
                for k in range(len(self.means)):
                    sub_k = np.subtract(x[i], self.means[k])
                    exp_t1 = np.matmul(-0.5 * sub_k,
                                       np.linalg.inv(self.variances[k]))
                    exp_t2 = np.matmul(exp_t1, sub_k.T)
                    r[i][k] = coef_k * np.exp(exp_t2) / sum_k

            # M step
            means = []
            variances = []
            pi = []
            for k in range(len(self.means)):
                n_k = np.sum(r[:, k])
                sub_k = np.subtract(x, self.means[k])
                r_k = r[:, k].reshape((N, 1))
                means.append(np.sum(np.multiply(r_k, x), axis=0) / n_k)
                var_n_t1 = np.multiply(r_k, sub_k)
                var_n_t2 = np.matmul(var_n_t1.T, sub_k)
                variances.append(var_n_t2 / n_k)
                pi.append(n_k / N)
            self.means = np.array(means)
            self.variances = np.array(variances)
            self.pi_k = np.array(pi)
            ll_x = self.compute_log_likelihood(x)
            if np.absolute(ll_x_old - ll_x) <= self.e:
                break
            ll_x_old = ll_x
            cnt += 1
        '''raise Exception('Implement fit function (filename: gmm.py)')'''
        return cnt
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE

        #raise Exception('Implement sample function in gmm.py')
        samples = []
        self.pi_k /= np.sum(self.pi_k)
        k_arr = np.random.choice(self.n_cluster, size=N, p=self.pi_k)
        for k in k_arr:
            samples.append(
                np.random.multivariate_normal(self.means[k], self.variances[
                    k]))
        return np.array(samples)
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement compute_log_likelihood function in gmm.py')

        #Check if variance matrix is invertible
        for k in range(len(self.means)):
            while True:
                if np.linalg.matrix_rank(
                        self.variances[k]) != self.variances[k].shape[0]:
                    self.variances[k] += 0.001 * np.identity(
                        self.variances[k].shape[0])
                else:
                    break
        sum_k = 0.0
        for i in range(len(x)):
            sum_i_k = 0.0
            for k in range(len(self.means)):
                coef_k = (1.0 / np.sqrt(
                    np.power(2 * np.pi, x.shape[1]) * np.linalg.det(
                        self.variances[k])))
                sub_k = np.subtract(x[i], self.means[k])
                exp_t1 = np.matmul(-0.5 * sub_k,
                                   np.linalg.inv(self.variances[k]))
                exp_t2 = np.matmul(exp_t1, sub_k.T)
                sum_i_k += coef_k * np.exp(exp_t2)
            if sum_i_k != 0.0:
                sum_k += np.log(sum_i_k)
        return float(sum_k)
        # DONOT MODIFY CODE BELOW THIS LINE
