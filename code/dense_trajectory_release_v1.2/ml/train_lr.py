#train a regressor
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, normalize
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
	(reg_type, reg_param) = args
	#define the model
	model = LogisticRegression(penalty=reg_type, C=reg_param)
	#get score from cv on training data
	scores = cross_validation.cross_val_score(model, data['features'], data['labels'], cv=3)
	error_rate = 1-scores.mean()
	return error_rate #minimise the error rate

#set the space
space = [hp.choice('reg', ['l1', 'l2']), hp.uniform('reg_param', 0, 100)]
	
#search
trials = Trials()

#prepare the data
data = {}
X = np.array(dt.feature_dict['Trajectory']); X = normalize(X, axis=1, norm='l2')
y = dt.feature_dict['Label']; y = [y[el][0] for el in range(len(y))]; y = np.array(y)
le = LabelEncoder()
y_t = le.fit_transform(y)
data['features'] = X
data['labels'] = y_t

best = fmin(fn=lambda args: objective(args,data), space=space, algo=tpe.suggest, max_evals=200, trials=trials)
loss_hist = [trials.trials[t]['result']['loss'] for t in range(len(trials.trials))]
#report results
print best
#print trials.trials

'''
feature = 'Trajectory' #try only with trajectory to begin
X = np.array(dt.feature_dict[feature])
'''
