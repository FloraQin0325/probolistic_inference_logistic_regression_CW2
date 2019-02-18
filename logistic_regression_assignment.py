import numpy as np
import numpy.random as rn
from scipy import optimize, stats
import scipy.linalg as linalg


# ##############################################################################
# load_data generates a binary dataset for visualisation and testing using two
# parameters:
# * A **jitter** parameter that controls how noisy the data are; and
# * An **offset** parameter that controls the separation between the two classes.
#
# Do not change this function!
# ##############################################################################
def load_data(N = 50, jitter=0.7, offset=1.2):
    # Generate the data
    x = np.vstack([rn.normal(0, jitter, (N // 2, 1)),
                   rn.normal(offset, jitter, (N // 2, 1))])
    y = np.vstack([np.zeros((N // 2, 1)), np.ones((N // 2, 1))])
    x_test = np.linspace(-2, offset + 2).reshape(-1, 1)

    # Make the augmented data matrix by adding a column of ones
    x_train = np.hstack([np.ones((N, 1)), x])
    x_test = np.hstack([np.ones((N, 1)), x_test])
    return x_train, y, x_test


# ##############################################################################
# predict takes a input matrix X and parameters of the logistic regression theta
# and predicts the output of the logistic regression.
# ##############################################################################
def predict(X, theta):
    # X: K x D matrix of test inputs
    # theta: D x 1 vector of parameters
    # returns: prediction of f(X); K x 1 vector
    prediction = np.zeros((X.shape[0], 1))

    # Task 1:
    # TODO: Implement the prediction of a logistic regression here.

    prediction = 1 / (1 + np.exp(-(X.dot(theta))))
    return prediction


def predict_binary(X, theta):
    # X: K x D matrix of test inputs
    # theta: D x 1 vector of parameters
    # returns: binary prediction of f(X); K x 1 vector; should be 0 or 1

    prediction = 1. * (predict(X, theta) > 0.5)

    return prediction


# ##############################################################################
# log_likelihood takes data matrices x and y and parameters of the logistic
# regression theta and returns the log likelihood of the data given the logistic
# regression.
# ##############################################################################
def log_likelihood(X, y, theta):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # theta: parameters (D x 1)
    # returns: log likelihood, scalar

    L = 0

    # Task 2:
    # TODO: Calculate the log-likelihood of a dataset
    # given a value of theta.

    for i in range(len(y)):
        sigma = X[i].dot(theta)
        sigma = 1 / (1 + np.exp(-sigma))
        L += y[i]*np.log(sigma) + (1-y[i])*np.log(1-sigma)

    return L.item()


# ##############################################################################
# max_lik_estimate takes data matrices x and y ands return the maximum
# likelihood parameters of a logistic regression.
# ##############################################################################
def max_lik_estimate(X, y):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # returns: maximum likelihood parameters (D x 1)

    N, D = X.shape

    theta_init = rn.rand(D, 1)
    theta_ml = theta_init

    # Task 3:
    # TODO: Optimize the log-likelihood function you've
    # written above an obtain a maximum likelihood estimate


    sigma_Zn = 1 / (1 + np.exp(-X.dot(theta_init)))

    grad_NLL = (sigma_Zn - y).T.dot(X).T

    max_iters = 50
    lr = 0.01
    epsilon = 1e-6
    i=0
    while (np.all(grad_NLL)!= 0):

        theta_ml_next = theta_ml  - lr * grad_NLL

        sigma_Zn = 1 / (1 + np.exp(-X.dot(theta_ml_next)))
        grad_NLL = (sigma_Zn - y).T.dot(X).T
        i += 1
        theta_ml = theta_ml_next
        if (i > 100):
            break
    return theta_ml


# ##############################################################################
# neg_log_posterior takes data matrices x and y and parameters of the logistic
# regression theta as well as a prior mean m and covariance S and returns the
# negative log posterior of the data given the logistic regression.
# ##############################################################################
def neg_log_posterior(theta, X, y, m, S):
    # theta: D x 1 matrix of parameters
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: scalar negative log posterior

    negative_log_posterior = 0

    # Task 4:
    # TODO: Calculate the log-posterior
    length = len(theta)

    log_l = log_likelihood(X, y, theta)
    log_prior =  - 0.5 * (theta - m).T.dot(np.linalg.inv(S)).dot(theta - m)
    negative_log_posterior -= (log_l + log_prior)
    return negative_log_posterior.item()


# ##############################################################################
# map_estimate takes data matrices x and y as well as a prior mean m and
# covariance  and returns the maximum a posteriori parameters of a logistic
# regression.
# ##############################################################################
def map_estimate(X, y, m, S):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: maximum a posteriori parameters (D x 1)

    N, D = X.shape

    theta_init = rn.rand(D, 1)
    theta_map = theta_init

    # Task 5:
    # TODO: Optimize the log-posterior function you've
    # written above an obtain a maximum a posteriori estimate

    # sigma_Zn = 1 / (1 + np.exp(-X.dot(theta_init)))
    # grad_NLP = np.linalg.inv(S).dot(theta_init-m) + (sigma_Zn - y).T.dot(X).T
    #
    # max_iters = 50
    # lr = 0.01
    # epsilon = 1e-6
    # i = 0
    # while (np.all(grad_NLP) != 0):
    #
    #     theta_map_next = theta_map - lr * grad_NLP
    #
    #     sigma_Zn = 1 / (1 + np.exp(-X.dot(theta_map_next)))
    #     grad_NLP = np.linalg.inv(S).dot(theta_map_next-m) + (sigma_Zn - y).T.dot(X).T
    #     i += 1
    #     theta_map = theta_map_next
    #     if (i > 100):
    #         break
    def func(theta):
        theta = theta.reshape(D, 1)
        return neg_log_posterior(theta, X, y, m, S)

    min = optimize.minimize(fun=func, x0=theta_init, method='BFGS')
    theta_map = min.x
    return theta_map


# ##############################################################################
# laplace_q takes an array of points z and returns an array with Laplace
# approximation q evaluated at all points in z.
# ##############################################################################
def laplace_q(z):
    # z: double array of size (T,)
    # returns: array with Laplace approximation q evaluated
    #          at all points in z

    q = np.zeros_like(z)

    # Task 6:
    # TODO: Evaluate the Laplace approximation $q(z)$.

    # mean  =2 variance =4
    # according to the gradient of the chi(x) to be 0 then calculate the x^*
    for i in range(len(z)):

        q[i] = (1/np.sqrt(2*np.pi*4))*np.exp(-(z[i]-2)**2/(2*4))

    return q


# ##############################################################################
# get_posterior takes data matrices x and y as well as a prior mean m and
# covariance and returns the maximum a posteriori solution to parameters
# of a logistic regression as well as the covariance approximated with the
# Laplace approximation.
# ##############################################################################
def get_posterior(X, y, m, S):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: maximum a posteriori parameters (D x 1)
    #          covariance of Laplace approximation (D x D)

    mu_post = np.zeros_like(m)
    S_post = np.zeros_like(S)

    # Task 7:
    # TODO: Calculate the Laplace approximation of p(theta | X, y)

    mu_post = map_estimate(X, y, m, S)
    theta_map = mu_post
    sigma_Zn = predict(X, theta_map)
    sigma_Zn = sigma_Zn.reshape(X.shape[0], 1)
    A = X.T.dot(sigma_Zn*(1-sigma_Zn)*X)
    S_post = np.linalg.inv(A + np.linalg.inv(S).T)
    return mu_post, S_post


# ##############################################################################
# metropolis_hastings_sample takes data matrices x and y as well as a prior mean
# m and covariance and the number of iterations of a sampling process.
# It returns the sampling chain of the parameters of the logistic regression
# using the Metropolis algorithm.
# ##############################################################################

def metropolis_hastings_sample(X, y, m, S, nb_iter):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: nb_iter x D matrix of posterior samples

    D = X.shape[1]
    samples = np.zeros((nb_iter, D))

    # Task 8:
    # TODO: Write a function to sample from the posterior of the
    # parameters of the logistic regression p(theta | X, y) using the
    # Metropolis algorithm.

    mu_like = max_lik_estimate(X,y)
    mu_post, S_post = get_posterior(X, y, m, S)
    theta_init = np.zeros((D, 1))
    for i in range(nb_iter):

        samples[i] = theta_init.T
        theta_new = np.random.multivariate_normal(theta_init.squeeze(), S)
        theta_new = theta_new.reshape(theta_new.shape[0], -1)

        p_pos_new = np.exp(-neg_log_posterior(theta_new, X, y, m, S))
        p_pos = np.exp(-neg_log_posterior(theta_init, X, y, m, S))
        u = np.random.uniform(0, 1)

        if p_pos_new >= p_pos * u:
            theta_init = theta_new
        else:
            theta_init = theta_init

    return samples


# #
# import matplotlib.pyplot as plt
#
# x, y, x_test = load_data()
# D = x.shape[1]
# nb_iter = 10000
# m = np.zeros((D, 1))
# S = 5*np.eye(D)
# nb_samples = 5
#
# theta_map, S_post = get_posterior(x, y, m, S)
# plt.scatter(x[:,1], y)
# for i in range(nb_samples):
#     th = np.random.multivariate_normal(theta_map.squeeze(), S_post)
#     plt.plot(x_test[:,1], predict(x_test, th))
# plt.show()
