from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam, sgd
from autograd import scipy as sp
import autograd.numpy.random as npr
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import sys


def get_bayes_lr_posterior(prior_var, noise_var, x_matrix, y_matrix, samples=100):
    '''Generates posterior samples for Bayesian linear regression model coefficients'''
    prior_variance = np.diag(prior_var * np.ones(x_matrix.shape[1])) # make it 2 D
    prior_precision = np.linalg.inv(prior_variance)

    joint_precision = prior_precision + x_matrix.T.dot(x_matrix) / noise_var 
    joint_variance = np.linalg.inv(joint_precision)
    joint_mean = joint_variance.dot(x_matrix.T.dot(y_matrix)) / noise_var

    #sampling 100 points from the posterior
    posterior_samples = np.random.multivariate_normal(joint_mean.flatten(), joint_variance, size=samples)

    return posterior_samples

def get_bayes_lr_posterior_predictives(noise_var,posterior_samples,x_test_matrix,samples = 100):
    '''Generates posterior predictions and posterior predictive samples given 
    Bayesian linear regression model coefficent posterior distribution.
    
    posterior_predictions: predicts each X_test obs on each sample coefficient from posterior_samples
    posterior_predictive_samples: samples from each posterior_prediction distribution, randomness comes from noise var

    '''
    posterior_predictions = np.dot(posterior_samples, x_test_matrix.T) 
    posterior_predictive_samples = posterior_predictions[np.newaxis, :, :] + np.random.normal(0, noise_var**0.5, size=(samples, posterior_predictions.shape[0], posterior_predictions.shape[1]))
    posterior_predictive_samples = posterior_predictive_samples.reshape((samples * posterior_predictions.shape[0], posterior_predictions.shape[1]))
    return posterior_predictions, posterior_predictive_samples

def viz_pp_samples(x_train,y_train,x_test,posterior_predictive_samples,title):
    # Compute the 97.5 th percentile of the posterior predictive predictions
    pp_upper = np.percentile(posterior_predictive_samples, 97.5, axis=0)

    # Compute the 2.5 th percentile of the posterior predictive predictions
    pp_lower = np.percentile(posterior_predictive_samples, 2.5, axis=0)

    # Compute the 50 th percentile of the posterior predictive predictions
    pp_mean = np.mean(posterior_predictive_samples, axis=0)
    
    plt.plot(x_test, pp_mean, color='red') # visualize the mean of the posterior predictive
    plt.fill_between(x_test, pp_upper, pp_lower, color='red', alpha=0.4, label='95% Conf. Interval') # visualize the 95% posterior predictive interval
    plt.scatter(x_train, y_train, color='black', label='training data') # visualize the training data
    plt.legend()
    plt.title(title)
    plt.ylim([0.9*y_train.min(), 1.1*y_train.max()])
    plt.show()

def get_percentile_interval(posterior_predictive_samples):
    # Compute the 97.5 th percentile of the posterior predictive predictions
    pp_upper = np.percentile(posterior_predictive_samples, 97.5, axis=0)

    # Compute the 2.5 th percentile of the posterior predictive predictions
    pp_lower = np.percentile(posterior_predictive_samples, 2.5, axis=0)

    # Compute the 50 th percentile of the posterior predictive predictions
    pp_mean = np.mean(posterior_predictive_samples, axis=0).reshape((1,-1))

    return pp_lower, pp_mean, pp_upper