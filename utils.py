
import matplotlib.pyplot as plt
import numpy as np
import config
import random

#function relating x and y
default_gen_func = lambda x: (x**3)

def generate_data(random_seed,
                  number_of_points=config.training_sample_size,
                noise_variance=config.y_noise_variance,
                region_size = 2,
                gap_size=4,
                boundary_size = 1,
                f = default_gen_func):
    '''
    Function for generating toy regression data


    Generates data from the function y = f(x) + epsilon = x**3 + epsilon with epsilon~N(0, config.y_noise_variance) by default
    with a gap (default size = 1) in the middle of the train set and a boundary (default size is 2) at the left & right extremes for the test set

    '''
    random.seed(random_seed)
    x_train_min = - gap_size/2 - region_size
    x_train_max = gap_size/2 + region_size
    #training x
    x_train = np.hstack((np.linspace(x_train_min, -gap_size/2, number_of_points//2), np.linspace(gap_size/2, x_train_max, number_of_points//2)))


    #y is equal to f(x) plus gaussian noise
    y_train = f(x_train) + np.random.normal(0, noise_variance**0.5, number_of_points)

    x_test = np.linspace(x_train_min - boundary_size, x_train_max + boundary_size, number_of_points)
    return x_train.reshape(1, -1), y_train.reshape(1, -1), x_test.reshape(1, -1)


def run_toy_nn(nn_model,architecture,params,random,x_train,y_train,x_test):
    #instantiate a Feedforward neural network object
    nn = nn_model(architecture, random=random)

    #fit my neural network to minimize MSE on the given data
    nn.fit(x_train, y_train, params)

    #predict on the test x-values
    y_test_pred = nn.forward(nn.weights, x_test)
    print(x_test.flatten().reshape(-1,1).shape)

    #visualize the function learned by the neural network
    plt.scatter(x_train.flatten(), y_train.flatten(), color='black', label='data')
    plt.plot(x_test.flatten(), y_test_pred.flatten(), color='red', label='learned neural network function')
    plt.legend(loc='best')
    plt.show()

def luna_snap(luna,iters,x_test,x_train,y_train):
    fig,ax = plt.subplots(len(iters),2 ,figsize=(10,20))
    c = 0
    for i in iters:

        # get weights at iter
        i_weights = luna.ff.weight_trace[i].reshape(1,-1)

        ##############
        # find auxilary output at iter
        ###############
        aux_y = luna.ff.forward(i_weights,x_test)[0]

        #plot each function
        for j in range(aux_y.shape[0]):
            ax[c][0].plot(x_test.flatten(),aux_y[j].flatten(), c = 'blue', alpha = 0.4)
            ax[c][0].set_title(f"Iter = {i}")

        ###############
        # find posterior predictive samples at iter
        ###############
        # Transform X with Feature Map for Bayes Reg
        final_layer_train = luna.ff.forward(i_weights, x_train, final_layer_out=True)

        # Conduct Bayes Reg on Final Layer
        i_posterior_samples =bh.get_bayes_lr_posterior(luna.prior_var,
                                                luna.y_noise_var,
                                                final_layer_train.T[:,:,0],
                                                y_train.T,
                                                samples=100)
        
        # get posterior predictives
        final_layer_test = luna.ff.forward(i_weights, x_test, final_layer_out=True)

        i_predictives, i_predictive_samples = bh.get_bayes_lr_predictives(luna.y_noise_var,
                                                i_posterior_samples,
                                                final_layer_test.T[:,:,0],
                                                n=100)

        ax[c][1] = bh.viz_pp_samples(x_train, y_train,x_test.flatten(),i_predictive_samples,"", ax[c][1])

        c+=1
    fig.savefig("scratch/luna_training.png")
