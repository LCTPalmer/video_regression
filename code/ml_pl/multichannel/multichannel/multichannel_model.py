from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from sklearn.metrics.pairwise import chi2_kernel #for default multichannel kernel
import numpy as np

## bug TODO
# leaving kpt as None leads to much longer computation, in comparison to feeding
# in the DEFAULT value!
#--------------------------------------------------------------------------

class MultiChannelModel(object):
    '''
    Scikit-learn style class for multichannel regression. Implements fit, 
    predict, and score methods.

    The multichannel data is taken as a tuple of 2D numpy arrays, where each
    channel array contains the same number of observations and each row in
    each channel array corresponds to the same observation (channels may have
    a different number of columns/features).

    Initialisation arguments:

    num_channels - int:
        the number of channels of the data

    model - instance of sklearn model:
        e.g. SVR(kernel='precomputed') (which is default)

    kernel_param_tuple - tuple of dicts of length num_channels, e.g.:

         kpt = (

               {'kernel_func': func_handle1,
                'param_dict': {'param_1': 0.5,
                               'param_2': 1.0}
               },

               {'kernel_func': func_handle2,
                'param_dict': {'param_1': 0.75,
                               'param_2': 0.6}
               },

                 ... [one dict for each channel]

               )

        default input value is None. If left as None, the kernel is initialised inside
        the MultiChannelSVR class as a tuple of default chi2_kernels of length num_channels.

        Can also pass as a single dictionary, e.g. kpt = {'kernel_func': rbf_kernel,
                                                          'param_dict': {'gamma': 2}}
        which will be duplicated into a tuple of length num_channels when the object is
        instantiated (i.e. each channel will use the same kernel).

    Methods:

    -----------------------------------------------------------------
    .fit(X, y, [fit_params])

    Inputs:

    X - tuple of numpy arrays:
        Length of X must be equal to num_channels e.g. (C1, C2, ..., Cnum_channels). Each
        numpy array, Ci, must have the same number of rows (observations), but the
        dimensionality of each channel (columns) is independent.

    y - list or numpy array:
        1d list/array of observation labels.

    fit_params - dict:
        optional parameter dictionary to pass to the underlying model fit method.


    Outputs:
        None

    ------------------------------------------------------------------
    .predict(X, [predict_params])

    Inputs:

    X - as for fit().

    predict_params - dict:
        optional parameter dictionary to pass to the underlying SVR predict method.

    Outputs:

    y_pred - numpy array:
        1d array of model predictions, equal in length to the number of observations in X[i]

    ------------------------------------------------------------------
    .score(X, y, [score_params])

    Inputs:

    X - as for fit().

    y - as for fit().

    score_params - dict:
        optional parameter dictionary to pass to the underlying SVR score method.

    Outputs:

    score - scalar:
        score value of model prediction (relative to 'true' value supplied as y).

    ------------------------------------------------------------------
    Usage

    X_train = (np.random.random((100,50)), np.random.random((100,50))) #2-channel data
    y_train = np.random.random((100,1))

    kpt = ({'kernel_func': chi2_kernel, 'param_dict': {'gamma': 1.5}},
           {'kernel_func': rbf_kernel, 'param_dict': {'gamma': 0.5}})

    underlying_model = sklearn.linear_model.Ridge(alpha=1) 

    clf = MultiChannelModel(num_channels=2, model=underlying_model, kernel_param_tuple=kpt)
    clf.fit(X_train, y_train)

    X_test = ... 2-channel data
    y_pred = clf.predict(X_test)

    '''

    def __init__(self, num_channels, model, kernel_param_tuple=None, channel_weights=None):

        self.num_channels = num_channels
        self.model = model

        #set up the channel weights
        if channel_weights is None:
            self.channel_weights = np.ones(num_channels)
        else:
            self.channel_weights = np.array(channel_weights)
        if self.channel_weights.ndim > 1:
            self.channel_weights = np.squeeze(self.channel_weights)

        #setup the kernel dictionary
        if isinstance(kernel_param_tuple, dict):
            kernel_param_tuple = tuple([kernel_param_tuple] * self.num_channels)
        if kernel_param_tuple is None:
            self.kernel_param_tuple = \
            tuple([ {'kernel_func': chi2_kernel, 'param_dict': {}} ] * self.num_channels) #default
        else:
            self.kernel_param_tuple = kernel_param_tuple

        #checks
        assert isinstance(self.num_channels, int)
        assert self.channel_weights.ndim == 1
        assert len(self.channel_weights) == self.num_channels
        assert len(self.kernel_param_tuple) == num_channels
        import inspect
        for c in self.kernel_param_tuple:
            inspect.isfunction(c['kernel_func'])
            isinstance(c['param_dict'], dict)

    #multichannel kernel computation
    def multichannel_kern(self, A, B, method_type):
        #checks
        assert method_type == 'fit' or method_type == 'predict'
        assert len(A) == len(B) == self.num_channels
        assert isinstance(A, tuple)
        assert isinstance(B, tuple)

        if method_type == 'fit':
            self.scale_fac = [] #initialise scale factor list

        #initialise the 3D gram array
        n_A = A[0].shape[0] # number of observations
        n_B = B[0].shape[0]
        gram_array = np.empty((n_A, n_B, self.num_channels))

        #fill the array
        for ii, (c_A, c_B, c_kern_params) in enumerate(zip(A,B,self.kernel_param_tuple)): #c_A ~ channel in A

            #raw gram matrix for this channel
            k = c_kern_params['kernel_func'](c_A, c_B, **c_kern_params['param_dict'])
            #k = -np.log(k) # take the natural log to undo the per channel exponential raise

            if method_type == 'fit': # need to build the scale_fac

                #compute the scale factor
                current_scale_fac = np.mean(k, axis=None)
                self.scale_fac.append(current_scale_fac)

                #add to the array
                gram_array[:,:,ii] = k / current_scale_fac

            elif method_type == 'predict':
                gram_array[:,:,ii] = k / self.scale_fac[ii]

        #make scale_fac immutable
        if method_type == 'fit':
            self.scale_fac = tuple(self.scale_fac)

        #take weighted average across channels
        weighted_gram = np.average(gram_array, axis=2, weights=self.channel_weights)
        return self.num_channels * weighted_gram # return the weighted SUM
        #return np.exp(-self.num_channels * weighted_gram) # take exponent if following original form

    #fit the model
    def fit(self, X, y, fit_params={}):
        K_train = self.multichannel_kern(X, X, method_type='fit') #calc the gram matrix
        assert isinstance(self.scale_fac, tuple) #check attribute has been set and is a tuple
        self.training_examples = X #retain for prediction
        self.model.fit(K_train, y, **fit_params) #fit the model

    #predict labels for new observations
    def predict(self, X, predict_params={}):
        if not hasattr(self, 'scale_fac'):
            raise ValueError('gram matrix scaling factors not assigned - need to fit model')
        K_test = self.multichannel_kern(X, self.training_examples, method_type='predict')
        return self.model.predict(K_test, **predict_params)

    #R^2 between model.predict(X) and y
    def score(self, X, y, score_params={}):
        if not hasattr(self, 'scale_fac'):
            raise ValueError('gram matrix scaling factors not assigned - need to fit model')
        K_test = self.multichannel_kern(X, self.training_examples, method_type='predict')
        return self.model.score(K_test, y, **score_params)

#multichannel k-fold cross-validation
def multichannel_KFoldCV(model, X, y, n_folds=3, verbose=False, sample_weights=None):
    '''
    simple k-fold cross-validation function for MultiChannelModel instance.

    Usage:

    #setup the multichannel model
    model = sklearn.svc.SVR(kernel='precomputed', C=1) #SVR always needs precomputed kernel
    kpt = {'kernel_func': rbf_kernel, 'param_dict': {'gamma': 2}}
    clf = MultiChannelModel(num_channels=2, model=model, kernel_param_tuple=kpt)

    #get k-fold score on our trainnig data
    score_list = multichannel_KFoldCV(clf, X_train, y_train, n_folds=5)
    overall_clf_score = np.mean(score_list)

    '''

    #deal with sample weights-----------------------------
    num_obs = len(y)
    if sample_weights is None:
        sample_weights = np.ones(num_obs)

    #checks
    assert isinstance(X, tuple), 'multichannel input X must be a tuple'
    assert model.num_channels == len(X)
    assert isinstance(model, MultiChannelModel), 'model must be instance of MultiChannelModel class'

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

            #add the values to this split's channel list
            X_train.append(c_X_train)
            X_test.append(c_X_test)

        #turn into multichannel tuples
        X_train, X_test = tuple(X_train), tuple(X_test)

        #split the labels
        y_train, y_test = y[train_index], y[test_index]

        #split the weights 
        sample_weights_train, sample_weights_test = sample_weights[train_index], sample_weights[test_index]

        #fit the model
        model.fit(X_train, y_train, fit_params={'sample_weight': sample_weights_train})

        #score the model
        score = model.score(X_test, y_test)

        if verbose:
            print 'R^2 score on split {0}: {1}'.format(split, score)

        #append to the list
        score_list.append(score)

    #return the list of scores
    return score_list
