from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials


def objective(x):
    from sklearn.svm import SVR
    from sklearn.metrics.pairwise import linear_kernel
    from sklearn.externals import joblib
    from multichannel import MultiChannelModel, multichannel_KFoldCV, theano_rbf as rbf_kernel, theano_chi2 as chi2_kernel
    import numpy as np
    import csv, os, time

    #output the time taken
    t0 = time.time()

    def create_kpt(num_channels, gammas):
        kernel_param_list = []
        for channel in xrange(num_channels):
            if channel < 4:
                kdict = {'kernel_func': chi2_kernel, 'param_dict': {'gamma': gammas[channel]}}
            elif channel == 5:
                kdict = {'kernel_func': rbf_kernel, 'param_dict': {'gamma': gammas[channel]}}
            kernel_param_list.append(kdict)
        kernel_param_tuple = tuple(kernel_param_list)
        return kernel_param_tuple

    #load the dataset
    dataset_root = '/home/luke/projects/THE_dataset' #directory where features/labels kept
    train_path = os.path.join(dataset_root, 'train_set_wc3d.pkl')
    test_path = os.path.join(dataset_root, 'test_set_wc3d.pkl')
    X_train, y_train = joblib.load(train_path)
    X_test, y_test = joblib.load(test_path)

    #run the exp
    gammas = [x['traj_gamma'], x['hog_gamma'], x['hof_gamma'],
              x['mbhx_gamma'], x['mbhy_gamma'], x['c3d_gamma']]
    kpt = create_kpt(6, gammas)
    C = x['C']
    model = SVR(kernel='precomputed', C=C)
    mcm = MultiChannelModel(num_channels=6, model=model, kernel_param_tuple=kpt)
    scores = multichannel_KFoldCV(mcm, X_train, y_train, n_folds=3, verbose=False)
    loss = 1-np.mean(scores)
    eval_time = time.time()

    #logging
    with open('svr_nonlin_log.csv','a') as f:
        fc = csv.writer(f)
        row = [loss, eval_time,C] + gammas
        fc.writerow(row)

    print x, loss, 'time taken: {}'.format(time.time()-t0)
    return {'loss': loss, 'eval_time': eval_time}


svr_nonlin_space= {'C': hp.lognormal('svr_C', 0, 1.5),
                 'traj_gamma': hp.lognormal('traj_gamma', 0, 1),
                 'hog_gamma': hp.lognormal('hog_gamma', 0, 1),
                 'hof_gamma': hp.lognormal('hof_gamma', 0, 1),
                 'mbhx_gamma': hp.lognormal('mbhx_gamma', 0, 1),
                 'mbhy_gamma': hp.lognormal('mbhy_gamma', 0, 1),
                 'c3d_gamma': hp.lognormal('c3d_gamma', 0, 1)}

trials = MongoTrials('mongo://localhost:1234/svr_nonlin/jobs')
best = fmin(objective, space=svr_nonlin_space, trials=trials, algo=tpe.suggest, max_evals=500)
