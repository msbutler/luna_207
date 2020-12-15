# primary authors: michael and elaine 

from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam, sgd
from autograd import scipy as sp
import autograd.numpy.random as npr
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import sys
import math
import utils


def get_prior_samples(x_matrix, prior_var, prior_mean=0, samples=100):
    '''Generates prior samples for Bayesian linear regression model coefficients'''

    prior_variance = np.diag(prior_var * np.ones(x_matrix.shape[1])) # make it 2 D
    prior_mean = prior_mean*np.ones(x_matrix.shape[1])

    # sample weights from prior
    prior_samples = np.random.multivariate_normal(prior_mean, prior_variance, size=samples)

    return prior_samples

def get_bayes_lr_posterior(prior_var, noise_var, x_matrix, y_matrix, samples=100,return_params = False):
    '''Generates posterior samples for Bayesian linear regression model coefficients

    prior_var - a scalar indicating prior variance on coefficients of baysian linear model
    noise_var - a scalar indicating variance of \eps where \eps ~ N(0,noise_var*I)
    x_matrix - the training data used in the likilihood function (in an NLM it's often the final layer)
            - (# of obs) * (# of features from basis expansion, i.e. width of final layer of NN)
    y_matrix - output variable, should be an array (# of obs, )
    samples - number of samples to draw from posterior
    return_params - a bool for whether the function should return the paramaters of the postrior distribution or samples
    '''
    prior_variance = np.diag(prior_var * np.ones(x_matrix.shape[1])) # make it 2 D
    prior_precision = np.linalg.inv(prior_variance)

    joint_precision = prior_precision + x_matrix.T.dot(x_matrix) / noise_var
    joint_variance = np.linalg.inv(joint_precision)
    joint_mean = joint_variance.dot(x_matrix.T.dot(y_matrix)) / noise_var

    if return_params == True:
        return joint_mean.flatten(), joint_variance

    else:
        #sampling 100 points from the posterior
        posterior_samples = np.random.multivariate_normal(joint_mean.flatten(), joint_variance, size=samples)

        return posterior_samples

# samples either come from prior or posterior distribution
# predictions: predicts each X_test obs on each sample coefficient from samples
# predictive_samples: samples from each prediction distribution, randomness comes from noise var
def get_bayes_lr_predictives(noise_var,samples,x_test_matrix,n=100):

    # calculate predictions and predictive samples
    predictions = np.dot(samples, x_test_matrix.T)

    predictive_samples = predictions[np.newaxis, :, :] + np.random.normal(0, noise_var**0.5, size=(n, predictions.shape[0],predictions.shape[1]))
    predictive_samples = predictive_samples.reshape((n * predictions.shape[0], predictions.shape[1]))

    return predictions, predictive_samples

def viz_pp_samples(x_train,y_train,x_test,posterior_predictive_samples,title_text, multi=None, ylim = [-150,150],gen_func=utils.default_gen_func):
    
    if multi ==None:
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
    
    else:
        ax = multi
        
    # Compute the 97.5 th percentile of the posterior predictive predictions
    pp_upper = np.percentile(posterior_predictive_samples, 97.5, axis=0)

    # Compute the 2.5 th percentile of the posterior predictive predictions
    pp_lower = np.percentile(posterior_predictive_samples, 2.5, axis=0)

    # Compute the 50 th percentile of the posterior predictive predictions
    pp_mean = np.mean(posterior_predictive_samples, axis=0)

    ax.plot(x_test, pp_mean, color='blue', label = 'mean prediction') # visualize the mean of the posterior predictive
    ax.fill_between(x_test, pp_upper, pp_lower, color='blue', alpha=0.3, label='95% Conf. Interval') # visualize the 95% posterior predictive interval
    ax.scatter(x_train, y_train, color='red', label='training data') # visualize the training data
    
    ax.plot(x_test,[gen_func(x) for x in x_test], color = 'black', label = 'true function')
    ax.set_title(title_text)
    ax.set_ylim(ylim)
    
    if multi == None:
        ax.legend()
        return fig
    else:
        return ax
    



def bayes_lr_logl(prior_var,noise_var,x_train,y_train,x_test,y_test):
    '''
    NOT READY FOR MULTIVARIATE INPUT

    Generates log likilihood estimate Bayesian linear regression model

    prior_var - a scalar indicating prior variance on coefficients of baysian linear model
    noise_var - a scalar indicating variance of \eps where \eps ~ N(0,noise_var*I)
    x_train - the training data used for nueral net (in an NLM it's the final layer)
            - dimension: (# of obs) * (# of features from basis expansion, i.e. width of final layer of NN)
    y_train - output variable of training data, should be an array (# of obs, 1 )
    x_test - validation data drawn from same data generating process of training data, same dimension
    y_test - `` ``
    '''


    # get posterior mean and variance
    posterior_mean, posterior_variance = get_bayes_lr_posterior(prior_var,
                                                                noise_var,
                                                                x_train,
                                                                y_train,
                                                                samples = 100,
                                                                return_params = True)

    def y_pdf_eval(ym, xm, posterior_mean, posterior_variance, noise_var):
        # pdf that returns probability of y_m | x_m, training_data over all models of posterior
            # p(y_m |x_m, Data) ~ N(posterior_mean.T*x_m, noise_var+x_m.T*posterior_variance*x_m)
            # according to hw 2, for bayesian linear models

        # y_m and x_m constitute a single test observation

        mu = np.dot(posterior_mean,xm)
        var = noise_var + np.dot(xm,np.dot(xm,posterior_variance))

        #evaluate p at y
        p = sp.stats.norm.pdf(y, loc = mu,scale = var**0.5)

        return p

    log_probs = []

    #iterate over each obs
    for i in range(x_test.shape[1]):
        y =y_test[i]
        x = x_test[i]
        p_t = y_pdf_eval(y,x,posterior_mean,posterior_variance,noise_var)
        log_probs.append(math.log(p_t))

    return sum(log_probs)

def get_percentile_interval(posterior_predictive_samples):
    # Compute the 97.5 th percentile of the posterior predictive predictions
    pp_upper = np.percentile(posterior_predictive_samples, 97.5, axis=0)

    # Compute the 2.5 th percentile of the posterior predictive predictions
    pp_lower = np.percentile(posterior_predictive_samples, 2.5, axis=0)

    # Compute the 50 th percentile of the posterior predictive predictions
    pp_mean = np.mean(posterior_predictive_samples, axis=0).reshape((1,-1))

    return pp_lower, pp_mean, pp_upper
