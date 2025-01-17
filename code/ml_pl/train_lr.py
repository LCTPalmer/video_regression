#train a regressor
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, normalize
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from hyperopt import hp, fmin, tpe, space_eval, Trials
from hyperopt.pyll import scope
import os

#load datasets
dataset_file = './dataset/train.pkl' #since we're training the classifier
assert os.path.isfile(dataset_file), 'specified dataset file does not exist'
dt = joblib.load(dataset_file)


#experimenting with hyperopt

#the objective for hyperopt to minimise
def objective(args, data):
	print args
	model = args
	#get score from cv on training data
	scores = cross_validation.cross_val_score(model, data['features'], data['labels'], cv=3)
	error_rate = 1-scores.mean()
	neg_acc = -scores.mean()
	return neg_acc #error_rate #minimise the error rate

#scope classifiers (hyperopt can then instantiate them)
scope.define(LogisticRegression)
scope.define(SVC)

#set the space
space = hp.pchoice('estimator', [
	(0.5, scope.LogisticRegression(
		penalty=hp.choice('reg_type', ['l2', 'l2']), C=hp.lognormal('reg_param', 0, 1)
		)),
	(0.5, scope.SVC(kernel='rbf', C=hp.loguniform('svc_C', 0, 10)-1, gamma=hp.loguniform('svc_rbf_gamma', 0, 1)-1
		))
	])

	
#search
trials = Trials()

#prepare the data
data = {}
X = np.array(dt.feature_dict['Trajectory']); X = normalize(X, axis=1, norm='l2')
y = dt.feature_dict['Label']; y = [y[elem][0] for elem in range(len(y))]; y = np.array(y)
le = LabelEncoder()
y_t = le.fit_transform(y)
data['features'] = X
data['labels'] = y_t

best = fmin(fn=lambda args: objective(args,data), space=space, algo=tpe.suggest, max_evals=10, trials=trials)
loss_hist = [trials.trials[t]['result']['loss'] for t in range(len(trials.trials))]
#report results
print best
#print trials.trials

'''
feature = 'Trajectory' #try only with trajectory to begin
X = np.array(dt.feature_dict[feature])
'''
