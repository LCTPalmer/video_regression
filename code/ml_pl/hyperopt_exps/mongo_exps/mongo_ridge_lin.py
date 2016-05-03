from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials


def objective(input_dict):
    from sklearn.linear_model import Ridge
    from sklearn.metrics.pairwise import linear_kernel
    from sklearn.externals import joblib
    from multichannel import MultiChannelModel, multichannel_KFoldCV, theano_rbf as rbf_kernel, theano_chi2 as chi2_kernel
    import numpy as np
    import csv, os, time
    
    #load the dataset
    dataset_root = '/home/luke/projects/THE_dataset' #directory where features/labels kept
    train_path = os.path.join(dataset_root, 'train_set_wc3d.pkl')
    test_path = os.path.join(dataset_root, 'test_set_wc3d.pkl')
    X_train, y_train = joblib.load(train_path)
    X_test, y_test = joblib.load(test_path)
    
    #run the exp
    alpha = input_dict['alpha']
    kpt = {'kernel_func': linear_kernel, 'param_dict': {}}
    model = Ridge(alpha=alpha)
    mcm = MultiChannelModel(num_channels=6, model=model, kernel_param_tuple=kpt)
    scores = multichannel_KFoldCV(mcm, X_train, y_train, n_folds=3, verbose=False)
    loss = 1-np.mean(scores)
    eval_time = time.time()
    
    #logging
    with open('ridge_lin_log.csv','a') as f:
        fc = csv.writer(f)
        row = [loss, eval_time, alpha]
        fc.writerow(row)

    print loss, input_dict
    return {'loss': loss, 'eval_time': eval_time}


ridge_lin_space = {'alpha': hp.lognormal('ridge_alpha', 0, 1)}
trials = MongoTrials('mongo://localhost:1234/ridge_lin/jobs') 
best = fmin(objective, space=ridge_lin_space, trials=trials, algo=tpe.suggest, max_evals=1000)
