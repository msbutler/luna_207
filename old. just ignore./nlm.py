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
        self.ff.fit(X, Y, params)

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
    
    def predict(self,X_test):

        # forward pass up to last layer
        final_layer = self.ff.forward(self.ff.weights, X_test, final_layer_out=True)
        
        # get posterior predictives, posterior of final layer weights
        return bh.get_bayes_lr_posterior_predictives(self.y_noise_var,self.posterior_samples,final_layer.T[:,:,0])