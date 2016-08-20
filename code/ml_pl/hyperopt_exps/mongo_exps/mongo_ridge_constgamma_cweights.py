from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials


def objective(x):
    from sklearn.linear_model import Ridge
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
    #set the krnel gammas set from previous experiment
    gammas = [0.06386, 0.117941, 0.060457, 0.180092, 3405.482, 0.79326]
    kpt = create_kpt(6, gammas)

    #set the alpha level
    alpha = x['alpha']

    #set the channel weights
    cw= [x['traj_cw'], x['hog_cw'], x['hof_cw'],
         x['mbhx_cw'], x['mbhy_cw'], x['c3d_cw']]

    model = Ridge(alpha=alpha)
    mcm = MultiChannelModel(num_channels=6, model=model, kernel_param_tuple=kpt, channel_weights=cw)
    scores = multichannel_KFoldCV(mcm, X_train, y_train, n_folds=3, verbose=False)
    loss = 1-np.mean(scores)
    eval_time = time.time()

    print 'params: {0}, loss: {1}, time taken: {2}'.format(x, loss, time.time()-t0)
    return {'loss': loss, 'eval_time': eval_time}


ridge_nonlin_space= {'alpha': hp.lognormal('ridge_alpha', 0, 1.5),
                 'traj_cw': hp.uniform('traj_cw', 0, 1),
                 'hog_cw': hp.uniform('hog_cw', 0, 1),
                 'hof_cw': hp.uniform('hof_cw', 0, 1),
                 'mbhx_cw': hp.uniform('mbhx_cw', 0, 1),
                 'mbhy_cw': hp.uniform('mbhy_cw', 0, 1),
                 'c3d_cw': hp.uniform('c3d_cw', 0, 1)}



trials = MongoTrials('mongo://localhost:1234/ridge_constgamma_cweights/jobs')
best = fmin(objective, space=ridge_nonlin_space, trials=trials, algo=tpe.suggest, max_evals=500)
