#general
import unittest
import numpy as np
from sklearn.preprocessing import Normalizer

#chi2_kernel
from theano_chi2 import theano_chi2
from sklearn.metrics.pairwise import chi2_kernel

#multichannel svr
from multichannel_svr import MultiChannelSVR, multichannel_KFoldCV

class TestMultiChannelSVR(unittest.TestCase):

	def test_chi2_theano_equals_sklearn(self):

		tolerance = .0001
		X = np.random.random((100,100))
		Y = np.random.random((200,100))
		X[0,0] = 0 #check for zero entries in histograms - make sure not dividing by zero
		theano_K = theano_chi2(X,Y)
		sklearn_K = chi2_kernel(X,Y)
		diff = np.abs(theano_K-sklearn_K)
		num_above_tolerance = np.sum(diff>tolerance)

		self.assertTrue(num_above_tolerance == 0)

	def test_mcsvr(self):
		X_channel = np.array([[0.1,0.5],[0.5, 0.1],[0.1,0.6],[0.7, 0.1]])
		X_train = (X_channel, X_channel)
		X_test = (X_channel+.1, X_channel+.01)
		Y = np.array([1, 10, 1, 10])

		#set up the model
		mcsvr = MultiChannelSVR(num_channels=2)

		#train
		mcsvr.fit(X_train,Y)

		#predict
		Y_pred = mcsvr.predict(X_test)		

		#assertions
		self.assertTrue(Y.shape == (4,))

		#scores
		train_score = mcsvr.score(X_train,Y)
		test_score = mcsvr.score(X_test,Y)

		#assertions
		print 'train_score = {0}, test_score = {1}'.format(train_score, test_score)
		self.assertTrue(isinstance(test_score, float))
		self.assertTrue(isinstance(train_score, float))
		self.assertTrue(test_score < train_score)

	def test_kfold(self):

		#dummy data
		X_channel = np.array([[0.1,0.5],[0.5, 0.1],[0.1,0.6],[0.7, 0.1]]) #the folds are the same
		X_channel = np.vstack((X_channel, X_channel))
		X_train = (X_channel, X_channel+.01)

		Y = np.array([1, 10, 1, 10])
		Y = np.hstack((Y,Y))

		#set up the model
		mcsvr = MultiChannelSVR(num_channels=2)

		#run a CV
		score_list = multichannel_KFoldCV(mcsvr, X_train, Y, n_folds=2, normalize=True, verbose=True)

		#assertions
		print score_list
		self.assertTrue(len(score_list) == 2)
		self.assertTrue(score_list[0] == score_list[1] > .5)


if __name__ == '__main__':
	unittest.main()