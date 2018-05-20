import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
import os
import pandas as pd

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[0]                  
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]


 
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def load_dataset():
    trainx = []
    trainy = []
    cnt = 1
    for imgfolder in os.listdir('lfw/'):
        cnt += 1
        for filename in os.listdir('lfw/'+imgfolder):
            file = 'lfw/'+imgfolder+'/'+filename
            image = cv2.imread(file)
            trainx.append(image)
            trainy.append(cnt)
    trainx = np.asarray(trainx)
    trainy = np.asarray(trainy)
    #assert trainy.shape == 1
    trainy = np.reshape(trainy, (trainy.shape[0],1))
    #trainx = np.reshape(trainx,(trainx[0], trainx[3], trainx[1], trainx[2]))
    return trainx, trainy

