import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    
    X = tf.placeholder('float',shape=(None, n_H0,n_W0,n_C0))
    Y = tf.placeholder('float',shape=(None, n_y))
    
    return X, Y


def initialize_parameters():
    
    tf.set_random_seed(1)                             
    W1 = tf.get_variable("W1",[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0))
    W2 = tf.get_variable("W2",[2,2,8,16],initializer = tf.contrib.layers.xavier_initializer_conv2d(seed =0))
    parameters = {"W1": W1,
                  "W2": W2}
    return parameters
def forward_propagation(X, parameters):
	W1 = parameters['W1']
    W2 = parameters['W2']
    
    Z1 = tf.nn.conv2d(X,W1,[1,1,1,1],padding = "SAME")
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides= [1,8,8,1],padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2,[1,1,1,1],padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1],strides=[1,4,4,1],padding='SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2,num_outputs=128, activation_fn = None)
    Z4 = tf.nn.l2_normalize(Z3, axis = 1)

    return Z4


def compute_cost(Z3, Y):

    #print(Y.shape, Z3.shape)
    cost = triplet_semihard_loss(labels = Y, embeddings = Z3)
    cost = tf.reduce_mean(cost)
    return cost
