#primary authors: michael and elaine

from autograd import numpy as np

training_sample_size = 100
training_seed = 124
valid_seed = 32847

######################
### Model Params
######################
prior_variance = 1 # chosen in the paper. declared "reasonable". who decided this
y_noise_variance = 9 # needs to match what the dataset itself is

######################
### Feed Forward Params
######################

####  activation function ####
activation_fn_type = 'relu'
activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)

#### optimization parameters ####
opt_params = {'step_size':1e-3,
          'max_iteration':100,
          'random_restarts':1,
          'optimizer':'adam'}

random_seed = np.random.RandomState(0)

real_max_iteration = 3500

########################
#### LUNA SPECIFIC  ####
########################

    # takes about 2 minutes to run

luna_architecture = {'width': 50,
            'hidden_layers': 2,
            'input_dim': 1,
            'output_dim': 50, #number of auxiliary functions, # note, in NLM, below, we change to 1
            'activation_fn_type': activation_fn_type,
            'activation_fn_params': 'rate=1',
            'activation_fn': activation_fn}

regularization_param_luna = 1e-1 # in the paper they searched over 1e-3,...,1e3 and chose 1e-1 for regularization

similarity_param = 1e0 # in the paper they searched over 1e-3,...,1e3 and chose 1e0 for similarity

########################
#### NLM SPECIFIC  ####
########################

    # takes about a second to run

nlm_architecture = luna_architecture.copy()
nlm_architecture["output_dim"] = 1
regularization_param_nlm = 8.37 #they chose this in the paper, what a beautifully specific number
