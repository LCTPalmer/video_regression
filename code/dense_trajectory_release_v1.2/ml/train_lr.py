#train a regressor
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from hyperopt import hp, fmin, tpe, space_eval, Trials
import os

#load datasets
dataset_file = './dataset/train.pkl' #since we're training the classifier
assert os.path.isfile(dataset_file), 'specified dataset file does not exist'
dt = joblib.load(dataset_file)


#experimenting with hyperopt

#the objective for hyperopt to minimise
def objective(args, data):
	l2_param = args
	#define the model
	model = LogisticRegression(penalty='l2', C=l2_param)
	#get score from cv on training data
	scores = cross_validation.cross_val_score(model, data['features'], data['labels'], cv=3)
	error_rate = 1 - scores.mean()
	return error_rate #minimise the error rate

#set the space
space = [(hp.uniform('l2_param', 0, 1))]
	
#search
trials = Trials()
c = 10

#prepare the data
data = {}
X = np.array(dt.feature_dict['Trajectory']); X = normalize(X, axis=1, norm='l2')
print X.shape
y = dt.feature_dict['Label']; y = [y[el][0] for el in range(len(y))]; y = np.array(y)
print y.shape, y[:5]
data['features'] = X
data['labels'] = y

best = fmin(fn=lambda args: objective(args,data), space=space, algo=tpe.suggest, max_evals=2, trials=trials)

#report results
print best
print space_eval(space, best)
#print trials.trials

'''
feature = 'Trajectory' #try only with trajectory to begin
X = np.array(dt.feature_dict[feature])
'''
