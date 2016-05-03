#class that does ridge regression using numpy
#option to weight samples in the .fit() method
import numpy as np
#import pdb
class WeightedRidge(object):
    def __init__(self, alpha=0, fit_intercept=True):
        self.l2_lambda = alpha
        self.fit_intercept = fit_intercept

    def _add_intercept(self, X):
        #add a column of ones to data matrix
        intercept = np.ones((X.shape[0], 1)) 
        return np.hstack((intercept, X))

    def fit(self, X, y, sample_weight=None):

        #deal with weights-----------------------------
        num_obs = X.shape[0]
        if sample_weight is None:
            sample_weight = np.ones((num_obs,1))

        #add singleton dim
        if len(sample_weight.shape) == 1:
            sample_weight = sample_weight[:,None]

        #make column
        if np.array_equal(sample_weight.shape, np.array([1, num_obs])): #if row-vector
            sample_weight = sample_weight.reshape(-1,1)

        #add singleton dim to y (so that multiplies correctly with weights)
        if len(y.shape) == 1:
            y = y[:, None]
        assert y.shape[0] == num_obs

        assert sample_weight.shape[0] == num_obs

        #pdb.set_trace()
        if self.fit_intercept:
            print X.shape
            X = self._add_intercept(X)
            print X.shape

        #----------------------------------------------
        #weight the observations and labels
        X_p = X * sample_weight #row based multiplication
        y_p = y * sample_weight
        #-----------------------------------------------

        #solve normal equations
        XTX = np.dot(X_p.T, X_p)
        assert XTX.shape[0] == XTX.shape[1]
        XTX = XTX + self.l2_lambda*np.eye(XTX.shape[0])
        XTX_inv = np.linalg.inv(XTX)
        pseudo_inverse = np.dot(XTX_inv, X_p.T)
        self.coef_ = np.dot(pseudo_inverse, y_p)
        return self

    def predict(self, X):
        if self.fit_intercept:
            return np.dot(self._add_intercept(X), self.coef_)
        else:
            return np.dot(X, self.coef_)

    def score(self, X, y):
        y_pred = self.predict(X)
        corr = np.corrcoef(y_pred, y)
        R2 = corr**2
        return R2
