import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from extract_dt import trajectory_generator, trajectory_split, FeatureDict
from load_hollywood import load_data

#BoF params
n_clusters = 4000 
save_dir = './codebook' #where to save the codebooks

#video file list
hollywood_dir = '/home/lukep86/Downloads/hollywood' #directory of the dataset
dataset_type = 'train' #training set or test set
videos, frame_ranges, classes = load_data(hollywood_dir, dataset_type)
#videos = ['../test_sequences/low_res.ogv', '../test_sequences/low_res2.ogv']

#sampling strategy
#take ~100,000 trajectories -> prob per trajectory of 100,000/2,000,000 = 0.05
threshold = 0.05

#instantiate trajectory features object
dtf = FeatureDict()

for ii, (video, frame_range) in enumerate(zip(videos,frame_ranges), start=1):
    print 'processing video %i of %i' % (ii, len(videos))
    dtp = ['-S', frame_range[0], '-E', frame_range[1], '-W', '10']#start and end frames
    t = trajectory_generator(video=video, dtp_terminal=dtp) #instantiate generator
    trajectory = t.next()
    t_cnt = 0
    while trajectory:
        if np.random.rand() < threshold:
            dtf.add_trajectory(trajectory)
        t_cnt += 1
        trajectory = t.next()
    print 'contained %i trajectories' % t_cnt

#clustering
model_dict = {}
for feature in dtf.feature_dict:
	X = np.array(dtf.feature_dict[feature])
	print 'clustering %s features' % feature + ' of dimension: ',X.shape 
	model_dict[feature] = KMeans(n_clusters=n_clusters).fit(X)
	print 'done'

#save the models
if not os.path.isdir(save_dir):
	os.mkdir(save_dir)
joblib.dump(model_dict, save_dir + '/model_dict.pkl')
