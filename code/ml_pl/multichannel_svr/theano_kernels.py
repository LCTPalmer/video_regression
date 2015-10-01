# GPU/Theano implementations of common kernel functions
#
# Currently contains:
#                     Radial Basis Function (theano_rbf)
#                     Chi-Squared (theano_chi2)
#
# Requires: 
#           Theano
#           Numpy

import theano
import theano.tensor as T
import numpy as np

def theano_rbf(X, Y=None, gamma=.5):

    '''GPU implementation of radial basis function (RBF) kernel:

               K(x,y) = e ^ ( -gamma .* ||x-y||^2 )

    Optimised by expanding the ||x-y||^2 term and calculating dot 
    products on GPU.

    Tested against sklearn.metrics.pairwise.rbf_kernel(). For X = Y of 
    size(1500,4000) is ~6x faster using a GeForce 560 Ti than sklearn on
    Intel Xeon W3520 quadcore CPU. 

    ----------------------------------------------------------------------

    usage:  K = theano_rbf(X, [Y=None, gamma=.5])

    input:  X - numpy array 2D
                    MxN array containing M observations of N features

            Y - numpy array 2D
                    PxN array (note must have same number of columns as X)

            gamma - scalar
                    spread parameter for rbf kernel

    output: K - numpy array 2D
                    MxP array of pairwise RBF similarities between
                    observations in X and Y
    '''

    #set Y - if None -> X
    Y = Y or X

    #check dimensions
    assert X.shape[1] == Y.shape[1], 'X and Y must be of same dimension'

    #define the symbolic vars
    x = T.matrix('x')
    y = T.matrix('y')
    
    x_dot_y = T.dot(x,y.T)

    #compile the theano function
    f = theano.function(inputs=[x, y], outputs=x_dot_y)

    #calc the dot products on GPU    
    X = X.astype('Float32')
    Y = Y.astype('Float32')
    x_dot_y = f(X,Y).astype('Float64') #for ease with sklearn
    
    #x^2 and y^2 terms in the expansion
    #these ops generally better on CPU due to memory
    x2 = np.tile(np.sum(X**2, axis=1)[:,None], [1, Y.shape[0]])
    y2 = np.tile(np.sum(Y**2, axis=1)[None,:], [X.shape[0], 1])

    #add together terms, multiply by gamma, and take exponential
    K = np.exp(-gamma * (x2 + y2 - 2*x_dot_y))

    return K

def theano_chi2(X, Y=None, gamma=1):

    '''GPU implementation of exponential chi-squared kernel - often used in 
    kernel classification and regression with bag-of-features representations:

               K(x,y) = e ^ ( -gamma .* Sum[ (xi-yi)^2 / (xi+yi) ] )
                                         i

    Tested against sklearn.metrics.pairwise.chi2_kernel(). For X = Y of 
    size(1500,4000) is ~3.5x faster using a GeForce 560 Ti than sklearn on
    Intel Xeon W3520 quadcore CPU. 

    ----------------------------------------------------------------------
    
    usage:  K = theano_chi2(X, [Y=None, gamma=.5])

    input:  X - numpy array 2D
                    MxN array containing M observations of N features

            Y - numpy array 2D
                    PxN array (note must have same number of columns as X)

            gamma - scalar
                    spread parameter for chi-squared kernel

    output: K - numpy array 2D
                    MxP array of pairwise chi-squared distances between
                    observations in X and Y
    '''

    #set Y - if None -> X
    Y = Y or X

    #check dimensions
    assert X.shape[1] == Y.shape[1], 'X and Y must be of same dimension'

    #add epsilon to avoid division by 0
    X = X + 1e-15
    Y = Y + 1e-15
    
    #ensure float32 type
    X = X.astype('Float32')
    Y = Y.astype('Float32')
    
    #declare constant for tiling (within K_row_comp())
    TILING_REPS_CONST = Y.shape[0]

    #constant for spread param
    GAMMA_CONST = gamma

    def K_row_comp(x_vec, y_mat):
        '''calculates a row of the chi-squared gram matrix'''

        #add singleton dimension to vector for tiling
        x_vec_2d = x_vec[:,None]

        #tile the vector
        x_vec_tiled = T.tile(x_vec_2d, [1, TILING_REPS_CONST])

        #subtract y and square
        x_min_y = x_vec_tiled - y_mat.T
        x_min_y2 = x_min_y ** 2

        #x + y term
        x_plus_y = x_vec_tiled + y_mat.T

        #divide and sum
        raw_row = T.sum( x_min_y2 / x_plus_y , axis=0)

        #take exponential
        K_row = T.exp(-GAMMA_CONST * raw_row)

        return K_row

    #define the symbolic vars
    x = T.matrix('x')
    y = T.matrix('y')

    #scan over the rows of X
    result, updates = theano.scan(K_row_comp, 
                                  outputs_info = None,
                                  sequences = x,
                                  non_sequences = y)

    #compile the theano function
    f = theano.function(inputs=[x, y], outputs=result)

    #return
    return f(X,Y).astype('Float64') #for ease with sklearn