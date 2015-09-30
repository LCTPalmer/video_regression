# GPU implementation of radial basis function (RBF) kernel:
#
#            K(x,y) = e ^ ( -gamma .* ||x-y||^2 )
#
# Optimised by expanding the ||x-y||^2 term and calculating dot 
# products on GPU.
# 
# Tested against sklearn.metrics.pairwise.rbf_kernel(). For X = Y of 
# size(1500,4000) is ~4.2x faster using a GeForce 560 Ti than sklearn on
# Intel Xeon W3520 quadcore CPU. 
#
# ----------------------------------------------------------------------
#
# input: X - numpy array 2D
#                 MxN array containing M observations of N features
#
#        Y - numpy array 2D
#                 PxN array (note must have same number of columns as X)
#
#	 gamma - scalar
#		  spread parameter for rbf kernel
#
# output: K - numpy array 2D
#                 MxP array of pairwise RBF similarities between
#                 observations in X and Y

import theano
import theano.tensor as T
import numpy as np

def theano_rbf(X, Y, gamma=.5):
    #check dimensions
    assert X.shape[1] == Y.shape[1], 'X and Y must be of same dimension'

    #define the symbolic vars
    x = T.matrix('x')
    y = T.matrix('y')

    #scan over the rows of X, taking dot products
    result, updates = theano.scan(lambda x_vec, y_mat: T.dot(x_vec, y_mat.T), 
                                  outputs_info = None,
                                  sequences = x,
                                  non_sequences = y)

    #compile the theano function
    f = theano.function(inputs=[x, y], outputs=result)

    #calc the dot products on GPU    
    X = X.astype('Float32')
    Y = Y.astype('Float32')
    x_dot_y = f(X,Y).astype('Float64') #for ease with sklearn
    
    #x^2 and y^2 terms in the expansion
    x2 = np.tile(np.sum(X**2, axis=1)[:,None], [1, Y.shape[0]])
    y2 = np.tile(np.sum(Y**2, axis=1)[None,:], [X.shape[0], 1])

    #add together terms, multiply by gamma, and take exponential
    K = np.exp(-gamma * (x2 + y2 - 2*x_dot_y))

    return K
