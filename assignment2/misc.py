##
# Miscellaneous helper functions
##

import numpy as np
from scipy.special import expit

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####

    e = np.sqrt( float(6) / (m+n) )
    A0 = np.random.uniform(-e,e,(m,n))

    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0

def softmax(x):
	e = np.exp(x - np.max(x,axis=1,keepdims=True))
	x = e / np.sum(e,axis=1,keepdims=True)
	return x

def sigmoid(x): 
    x = expit(x)
    return x

def sigmoid_grad(f):
    f = f * (1 - f)    
    return f