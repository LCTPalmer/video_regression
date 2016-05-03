#general
import unittest
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR

#chi2_kernel
from sklearn.metrics.pairwise import chi2_kernel

#multichannel svr
from multichannel import MultiChannelModel, multichannel_KFoldCV

class TestMultiChannelModel(unittest.TestCase):

    def test_mcm(self):
        X_channel = np.array([[0.1,0.5],[0.5, 0.1],[0.1,0.6],[0.7, 0.1]])
        X_train = (X_channel, X_channel)
        X_test = (X_channel+.1, X_channel+.01)
        Y = np.array([1, 10, 1, 10])

        #set up the model
        model = SVR(kernel='precomputed', C=1)
        mcm = MultiChannelModel(num_channels=2, model=model)

        #train
        mcm.fit(X_train,Y)

        #predict
        Y_pred = mcm.predict(X_test)        

        #assertions
        self.assertTrue(Y.shape == (4,))

        #scores
        train_score = mcm.score(X_train,Y)
        test_score = mcm.score(X_test,Y)

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
        model = SVR(kernel='precomputed', C=1)
        mcm = MultiChannelModel(num_channels=2, model=model)

        #run a CV
        score_list = multichannel_KFoldCV(mcm, X_train, Y, n_folds=2, verbose=True)

        #assertions
        print score_list
        self.assertTrue(len(score_list) == 2)
        self.assertTrue(score_list[0] == score_list[1] > .5)


if __name__ == '__main__':
    unittest.main()
