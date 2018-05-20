import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
from tf_utils import *



trainx, trainy = load_dataset()

def model(X_train, Y_train,  learning_rate = 0.009,num_epochs = 100, minibatch_size = 64, print_cost = True):
                   
    tf.set_random_seed(1)                             
    seed = 3                                          
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = 1                           
    costs = []                                    
    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)

    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3,Y)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                
                _ , temp_cost = sess.run([optimizer,cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        return parameters


params = model(trainx, trainy)