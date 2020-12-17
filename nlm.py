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
from scipy.stats import multivariate_normal

#user imports
import sys
sys.path.append(".")
from feed_forward import Feedforward
import bayes_helpers as bh

class NLM():
    """
    Fits NLM Model

    Model Assumptions
     - Weights distributed normally
     - Ys distributed normally


     How to use:
      - run train() to create:
            a) the NN MLE weights, found in self.ff.weights
            b) self.posterior samples, the distribution for the weights in the last layer of NLM

      - run predict() to get distribution of ys, given x test
    """
    def __init__(self, prior_var, y_noise_var, regularization_param, architecture, random_state, objective_function=None):

        self.ff = Feedforward(architecture, random = random_state, objective_function=objective_function)
        self.regularization_param = regularization_param
        self.prior_var = prior_var # prior variance final layer weights
        self.y_noise_var = y_noise_var # variance of noise, distributed normally

    def train(self, X, Y, params):
        # Fit Weights
        self.ff.fit(X, Y, params, self.regularization_param)

        # Transform X with Feature Map for Bayes Reg
            # i.e. returns last layer by setting final_layer_out to True
        final_layer = self.ff.forward(self.ff.weights, X, final_layer_out=True)
        # Conduct Bayes Reg on Final Layer
        self.posterior_samples = bh.get_bayes_lr_posterior(self.prior_var,
                                                        self.y_noise_var,
                                                        final_layer.T[:,:,0],
                                                        Y.T,
                                                        samples=100)

        print("\nDone Training")

    def predict(self,X_test,prior=False):
        """ returns either posterior predictives or prior predictives"""

        # forward pass up to last layer
        final_layer = self.ff.forward(self.ff.weights, X_test, final_layer_out=True)


        # return prior predictives and prior predictive samples
        if prior:
            prior_samples = bh.get_prior_samples(final_layer.T[:,:,0], self.prior_var, samples=100)
            return bh.get_bayes_lr_predictives(self.y_noise_var,
                                                prior_samples,
                                                final_layer.T[:,:,0],
                                                n=100)

        # return posterior predictives and posterior predictive samples
        else:
            return  bh.get_bayes_lr_predictives(self.y_noise_var,
                                                self.posterior_samples,
                                                final_layer.T[:,:,0],
                                                n=100)

    def get_log_l(self,X_train,Y_train,X_test,Y_test):
        """ return log likilood of nueral linear model, equivalent to calculating
        log likilihood of bayesian linear model using final layer basis expansion as
        X data """

        # forward pass up to last layer
        X_test_fl = self.ff.forward(self.ff.weights, X_test, final_layer_out=True)
        X_train_fl = self.ff.forward(self.ff.weights, X_train, final_layer_out=True)

        # ensure all data matrices ares obs by features
        log_l = bh.bayes_lr_logl(self.prior_var,
                             self.y_noise_var,
                             X_train_fl.T[:,:,0],
                             Y_train.T,
                             X_test_fl.T[:,:,0],
                             Y_train.T)
        return log_l





if __name__ == "__main__":
    # test
    prior_var = .1
    y_var = 1.0
    regularization_param_nlm = 10
    test_nlm = NLM(prior_var,y_var, regularization_param_nlm,architecture, random_state = np.random.RandomState(0))

    params = {'step_size':1e-3,
          'max_iteration':500,
          'random_restarts':1,
          'optimizer':'adam'}

    t0 = time.time()
    test_nlm.train(x_train,y_train, params)
    nlm_time = np.round(time.time() - t0, 3)
    print(f"{nlm_time} Seconds")

    posterior_predictives, posterior_predictive_samples = test_nlm.predict(x_test)
