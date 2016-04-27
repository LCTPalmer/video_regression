# GPU implementation of exponential chi-squared kernel - often used in 
# kernel classification and regression with bag-of-features representations:
#
#            K(x,y) = e ^ ( -gamma .* Sum[ (xi-yi)^2 / (xi+yi) ] )
#                                      i
#
# Tested against sklearn.metrics.pairwise.chi2_kernel(). For X = Y of 
# size(1500,4000) is ~3.5x faster using a GeForce 560 Ti than sklearn on
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
#		  spread parameter for chi-squared kernel
#
# output: K - numpy array 2D
#                 MxP array of pairwise chi-squared distances between
#                 observations in X and Y

import theano
import theano.tensor as T
import numpy

def theano_chi2(X, Y, gamma=1):
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
