# Scikit-learn style class for multichannel support vector regression (SVR)
# with chi-squared kernel. Implements fit, predict, and score methods.
#
# The MultiChannelSVR class is initialised with the number of channels of
# the data, and an optional parameter dictionary to feed into the 
# underlying sklearn.svm.SVR model (see sklearn doc for details). 
# 
# The multichannel data is taken as a tuple of 2D numpy arrays, where each 
# channel array contains the same number of observations and each row in 
# each channel array corresponds to the same observation (channels may have 
# a different number of columns/features).
#
# Preferentially uses a theano GPU implementation of the chi-squared kernel,
# contained in 'theano_chi2.py', which should be kept in the same directory as
# this file.
#
# A function, multichannel_KFoldCV, is also included to conduct a k-fold cross-
# validation procedure on a supplied MultiChannelSVR() model instance.

from sklearn.svm import SVR
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import KFold
import numpy as np

#import theano implementation of chi-squared kernel if it exists
# try:
# 	from theano_kernels import theano_chi2 as chi2_kernel
# 	print 'Using Theano implementation of chi-squared kernel'
# except ImportError:
# 	from sklearn.metrics.pairwise import chi2_kernel
# 	print 'Using scikit-learn implementation of chi-squared kernel'

from sklearn.metrics.pairwise import chi2_kernel
#--------------------------------------------------------------------------

class MultiChannelSVR():

    def __init__(self, num_channels, param_dict={'C': 1}, gamma_tuple=None):
        self.num_channels = num_channels
        self.model = SVR(kernel='precomputed', **param_dict)
        self.gamma_tuple = gamma_tuple or tuple([1] * num_channels) #spread param for each channel
        assert len(self.gamma_tuple) == num_channels
    
    #multichannel kernel computation
    def multichannel_chi2(self, A, B, gram_type):
        assert gram_type == 'fit' or gram_type == 'predict'
        assert len(A) == len(B) == self.num_channels
        assert isinstance(A, tuple)
        assert isinstance(B, tuple)
        
        if gram_type == 'fit':
            self.scale_fac = [] #initialise scale factor list
        
        #initialise the 3D gram array
        n_A = A[0].shape[0] # number of observations
        n_B = B[0].shape[0]
        gram_array = np.empty((n_A, n_B, self.num_channels))
        
        #fill the array
        for ii, (c_A, c_B, c_gamma) in enumerate(zip(A,B,self.gamma_tuple)): #c_A ~ channel in A
            
            #raw gram matrix for this channel
            k = chi2_kernel(c_A, c_B, gamma=c_gamma)
            
            if gram_type == 'fit': # need to build the scale_fac
                
                #compute the scale factor
                current_scale_fac = np.mean(k, axis=None)
                self.scale_fac.append(current_scale_fac)
                
                #add to the array
                gram_array[:,:,ii] = k / current_scale_fac
                
            elif gram_type == 'predict':
                gram_array[:,:,ii] = k / self.scale_fac[ii]
        
        #make scale_fac immutable
        if gram_type == 'fit':
            self.scale_fac = tuple(self.scale_fac)
        
        return np.sum(gram_array, axis=2) # sum across channels          
   
    #fit the model
    def fit(self, X, y):
        K_train = self.multichannel_chi2(X, X, gram_type='fit') #calc the gram matrix
        assert isinstance(self.scale_fac, tuple) #check attribute has been set and is a tuple
        self.training_examples = X #retain for prediction
        self.model.fit(K_train, y) #fit the model
    
    #predict labels for new observations
    def predict(self, X):
        if not hasattr(self, 'scale_fac'):
            raise ValueError('gram matrix scaling factors not assigned - need to fit model')
        K_test = self.multichannel_chi2(X, self.training_examples, gram_type='predict')
        return self.model.predict(K_test)
    
    #R^2 between model.predict(X) and y
    def score(self, X, y):
        if not hasattr(self, 'scale_fac'):
            raise ValueError('gram matrix scaling factors not assigned - need to fit model')
        K_test = self.multichannel_chi2(X, self.training_examples, gram_type='predict')
        return self.model.score(K_test, y)


class MultiChannelSVR2():
    '''
    kernel_param_tuple is tuple of dicts of form: {'kernel_func': func_handle,
                                                  'param_dict': {'param_1': 0.5,
                                                                 'param_2': 1.0}
                                                  }
    '''

    def __init__(self, num_channels, model_param_dict={'C': 1}, kernel_param_tuple=None):
        self.num_channels = num_channels
        self.model_param_dict = model_param_dict
        self.model = SVR(kernel='precomputed', **self.model_param_dict)
        self.kernel_param_tuple = kernel_param_tuple or \
            tuple([ {'kernel_func': chi2_kernel, 'param_dict': {}} ] * num_channels) #default

        #checks
        assert len(self.kernel_param_tuple) == num_channels
        import inspect
        for c in kernel_param_tuple:
            inspect.isfunction(c['kernel_func'])
            isinstance(c['param_dict'], dict)
    
    #multichannel kernel computation
    def multichannel_chi2(self, A, B, gram_type):
        assert gram_type == 'fit' or gram_type == 'predict'
        assert len(A) == len(B) == self.num_channels
        assert isinstance(A, tuple)
        assert isinstance(B, tuple)
        
        if gram_type == 'fit':
            self.scale_fac = [] #initialise scale factor list
        
        #initialise the 3D gram array
        n_A = A[0].shape[0] # number of observations
        n_B = B[0].shape[0]
        gram_array = np.empty((n_A, n_B, self.num_channels))
        
        #fill the array
        for ii, (c_A, c_B, c_kern_params) in enumerate(zip(A,B,self.kernel_param_tuple)): #c_A ~ channel in A
            
            #raw gram matrix for this channel
            k = c_kern_params['kernel_func'](c_A, c_B, **c_kern_params['param_dict'])
            
            if gram_type == 'fit': # need to build the scale_fac
                
                #compute the scale factor
                current_scale_fac = np.mean(k, axis=None)
                self.scale_fac.append(current_scale_fac)
                
                #add to the array
                gram_array[:,:,ii] = k / current_scale_fac
                
            elif gram_type == 'predict':
                gram_array[:,:,ii] = k / self.scale_fac[ii]
        
        #make scale_fac immutable
        if gram_type == 'fit':
            self.scale_fac = tuple(self.scale_fac)
        
        return np.sum(gram_array, axis=2) # sum across channels          
   
    #fit the model
    def fit(self, X, y, sample_weight=None):
        K_train = self.multichannel_chi2(X, X, gram_type='fit') #calc the gram matrix
        assert isinstance(self.scale_fac, tuple) #check attribute has been set and is a tuple
        self.training_examples = X #retain for prediction
        self.model.fit(K_train, y, sample_weight=sample_weight) #fit the model
    
    #predict labels for new observations
    def predict(self, X):
        if not hasattr(self, 'scale_fac'):
            raise ValueError('gram matrix scaling factors not assigned - need to fit model')
        K_test = self.multichannel_chi2(X, self.training_examples, gram_type='predict')
        return self.model.predict(K_test)
    
    #R^2 between model.predict(X) and y
    def score(self, X, y):
        if not hasattr(self, 'scale_fac'):
            raise ValueError('gram matrix scaling factors not assigned - need to fit model')
        K_test = self.multichannel_chi2(X, self.training_examples, gram_type='predict')
        return self.model.score(K_test, y)

#multichannel k-fold cross-validation
def multichannel_KFoldCV(model, X, y, n_folds=3, normalize=True, verbose=False):
    #check data is multichannel
    assert isinstance(X, tuple), 'multichannel input X must be a tuple'
    #assert len(X)>1
    #check model is MultiChannelSVR
    assert isinstance(model, MultiChannelSVR), 'model must be instance of MultiChannelSVR class'
    
    #number of samples
    n_X = X[0].shape[0]
    
    #generate the k-folds
    kf = KFold(n_X, n_folds=n_folds)
    
    #initialise the score list for returning
    score_list = []
    
    #loop through splits of the data
    for split, (train_index, test_index) in enumerate(kf, start=1):
        
        if verbose:
            print 'training on {0} samples, testing on {1} samples'.format(len(train_index), len(test_index))
        
        #initialise multichannel lists of splits
        X_train, X_test = [], []
        
        for c_X in X: # c_X ~ channel in X
            
            #get the values for this split-channel combination
            c_X_train, c_X_test = c_X[train_index,:], c_X[test_index,:]

            #normalize
            if normalize:
            	normalizer = Normalizer(norm='l1')
                c_X_train = normalizer.fit_transform(c_X_train)
            	c_X_test = normalizer.transform(c_X_test)
            
            #add the values to this split's channel list
            X_train.append(c_X_train)
            X_test.append(c_X_test)
        
        #turn into multichannel tuples
        X_train, X_test = tuple(X_train), tuple(X_test)
        
        #split the labels
        y_train, y_test = y[train_index], y[test_index]
        
        #fit the model
        model.fit(X_train, y_train)
        
        #score the model
        score = model.score(X_test, y_test)
        
        if verbose:
            print 'R^2 score on split {0}: {1}'.format(split, score)
        
        #append to the list
        score_list.append(score)
    
    #return the list of scores
    return score_list   

#
def multichannel_KFoldCV_opt(X, y, model_params={'C': 1}, n_folds=3, normalize=True, verbose=False):
    #in this version - calculate the full gram matrix first and take slices of that
    #check data is multichannel
    assert isinstance(X, tuple), 'multichannel input X must be a tuple'
    #assert len(X)>1
    #check model is MultiChannelSVR
    assert isinstance(model, MultiChannelSVR), 'model must be instance of MultiChannelSVR class'

    K_train = chi2_kernel(X,X)
    
    #number of samples
    n_X = X[0].shape[0]
    
    #generate the k-folds
    kf = KFold(n_X, n_folds=n_folds)
    
    #initialise the score list for returning
    score_list = []
    
    #loop through splits of the data
    for split, (train_index, test_index) in enumerate(kf, start=1):
        
        if verbose:
            print 'training on {0} samples, testing on {1} samples'.format(len(train_index), len(test_index))
        
        #initialise multichannel lists of splits
        X_train, X_test = [], []
        
        for c_X in X: # c_X ~ channel in X
            
            #get the values for this split-channel combination
            c_X_train, c_X_test = c_X[train_index,:], c_X[test_index,:]

            #normalize
            if normalize:
                normalizer = Normalizer(norm='l1')
                c_X_train = normalizer.fit_transform(c_X_train)
                c_X_test = normalizer.transform(c_X_test)
            
            #add the values to this split's channel list
            X_train.append(c_X_train)
            X_test.append(c_X_test)
        
        #turn into multichannel tuples
        X_train, X_test = tuple(X_train), tuple(X_test)
        
        #split the labels
        y_train, y_test = y[train_index], y[test_index]
        
        #fit the model
        model.fit(X_train, y_train)
        
        #score the model
        score = model.score(X_test, y_test)
        
        if verbose:
            print 'R^2 score on split {0}: {1}'.format(split, score)
        
        #append to the list
        score_list.append(score)
    
    #return the list of scores
    return score_list   
