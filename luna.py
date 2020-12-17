



#Primary Authors: Michael and Elaine

# import standard libraries
from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam, sgd
from autograd import scipy as sp
import autograd.numpy.random as npr
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import sys

# import our libraries
# from utils import cos_sim_sq
from nlm import NLM



class LUNA(NLM):
    """
    Fits LUNA Model; inherits from NLM and overrides the objective function

    Model Assumptions
     - Weights distributed normally
     - Ys distributed normally

     How to use:
      - run train() to create:
            a) the NN MLE weights, found in self.ff.weights
            b) self.posterior samples, the distribution for the weights in the last layer of NLM

      - run predict() to get distribution of ys, given x test
    """


    def __init__(self, prior_var, y_noise_var, regularization_param, similarity_param, architecture, random, grad_func_specs = None):
        '''

        Input attributes:
         - prior_var: variance on weights for last layer (the bayesian part of the NLM)
         - y_noise_var: variance of epsilon, given our model is Y = wX + \eps where \eps ~ N(0,y_noise_var)
         - regularization_param: scalar to regularlize the NLM part of the objective function
         - architecture: specifies the feed forward archicture, and no other aspects of the model
         - random: sets a random state
         - grad_func_specs: dictionary to specify how to conduct default_finite_differences


        NN attributes pulled from the super().__init__:
            self.D = not exactly sure...
            self.D_in = dimensionality of input data
            self.D_out = dimensionality of output data
            self.H = width of a layer in the NN
        '''

        self.similarity_param = similarity_param

        #inherit from NLM, override objective func
        super().__init__(prior_var, y_noise_var, regularization_param, architecture, random, self.make_objective)

        self.D, self.D_in, self.D_out, self.H = self.ff.D, architecture['input_dim'], architecture['output_dim'], architecture['width']

        self.grad_func_specs = grad_func_specs

        self.mse_trace = []
        self.reg_trace = []
        self.sim_score_trace = []




    def similarity_score(self, W, x):
        '''
        Calculates total sum of squared cosine similarity between all pairwise combinations of aux
        functions

        Inputs:
        - W = NumPy array of weights [dim=(1, width H, input dimension D_in)]

        Returns:
        - score = total cosine similarity squared across all pairs of functions [scalar]

        '''

        #D_out = self.D_out
        score = 0

        #derivs of all the aux funcs
        holy_grail = self.default_finite_diff(W, x)

        # in dim x out dim x # obs
        M = holy_grail.shape[1]

        # conduct compare each pair of aux functions
        for i in range(self.D_out):
            grad_i = holy_grail[:,i,:]
            for j in range(i + 1, self.D_out):
                grad_j = holy_grail[:,j,:]
                score += self.cos_sim_sq(grad_i, grad_j)

        return score

    def cos_sim_sq(self,grad_i, grad_j):
        numer = np.dot(grad_i, grad_j.T)**2
        denom = (np.dot(grad_i,grad_i.T)*np.dot(grad_j,grad_j.T))
        return (numer/denom)[0][0]

    def default_finite_diff(self,W,x):
        '''
        x.shape[0] is # of dimensions == self.D_in
        x.shape[1] is # of observations

        output: Returns a 3d matrix:
                (in dimension) x (out dimension (# of aux functions)) x (# observations)

                i.e. for each auxillary function and for each observation, approximate the gradient with dimension x.shape[0]
        '''

        #create one epsilon for each observation
        eps = np.random.normal(0,0.1,size=x.shape[1])

        #iterate over features of raw input data (rows of x)
        out = np.zeros((self.D_in, self.D_out, x.shape[1]))

        #evaluate function at x
        f_ex = self.ff.forward(W, x)

        assert x.shape[0] == self.D_in

        #for one dimension at a time
        for i in range(x.shape[0]):

            delta = np.zeros(x.shape)
            delta[i,:] = eps

            f_eps = self.ff.forward(W,x+delta)

            # out dim X #obs
            res = (f_eps - f_ex)/eps
            #out[i,:,:] = res[0] # value wise division, different epsilon for each column
            # NEED TO FIX FOR MULTIDIMENSIONAL INPUT DATA

        return res

    def mean_mean_sq_error(self, W, x_train, y_train):
        '''
        Calculates average mean sq error across each output nodes (=the aux functions)

        Inputs:
        - W = NumPy array of all weights [dim=(1, width H, input dimension D_in)]

        Returns:
        - mean_mse = mean of mean sq error for each aux function [scalar]
        '''
        D, D_out, H = self.D, self.D_out, self.H
        aux_outputs = self.ff.forward(W, x_train)
        Y = np.tile(y_train, D_out).reshape(1, D_out, y_train.shape[1])

        # calculate squared error for each aux regressor, take mean
        mean_mse = np.mean(np.linalg.norm(Y - aux_outputs, axis=1)**2)

        return mean_mse

    def make_objective(self, x_train, y_train, reg_param):
        '''
        Makes objective function and gradient of obj function

        Inputs:
        - x_train = NumPy array of training data [dim=(1, anything)]
        - y_train = NumPy array of training data [dim=(1, anything)]
        - reg_param = regularization parameter [scalar]

        Returns:
        - objective = function handle for objective function
        - grad(objective) = Autograd gradient of objective function
        '''

        def objective(W, t):
            '''
            Calculates objective function: L_luna(model) = L_fit(model) - L_similarity(model)
            L_fit(model) = average mean sq error across all outputs/aux functions
            L_similarity(model) = sum of squared cosine similarity across all aux function combinations

            Inputs:
            - W = NumPy array of all weights [dim=(1, width H, input dimension D_in)]
            - t = necessary for adam solver in Autograd (something about creating a callback)

            Returns:
            - L_fit - L_sim = function handle for objective function [scalar]
            '''

            mean_mse = self.mean_mean_sq_error(W, x_train, y_train)
            reg_penalty = np.linalg.norm(W)**2
            sim_score = self.similarity_score(W, x_train)

            if isinstance(reg_penalty,float):
                self.mse_trace.append(mean_mse)
                self.reg_trace.append(reg_penalty)
                self.sim_score_trace.append(sim_score)



            return mean_mse + self.regularization_param*reg_penalty + self.similarity_param*sim_score

        return objective, grad(objective)
