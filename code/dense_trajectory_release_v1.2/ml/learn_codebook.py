import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from extract_dt import trajectory_generator, trajectory_split 

#video file list
videos = ['../test_sequences/low_res.ogv', '../test_sequences/low_res2.ogv']

#sampling strategy
#take 50 trajectories from each video -> prob per trajectory of 100,000/4,000,000 = 0.025
threshold = 0.025

#initialise trajectory features structure 
dt_features = ({'Trajectory': [],
                      'HOG': [],
                      'HOF': [],
                      'MBHx': [],
                      'MBHy': []
                      })

for ii,video in enumerate(videos):
    print 'processing video %i of %i' % (ii+1, len(videos))
    t = trajectory_generator(video=video) #instantiate generator
    line = t.next()
    while line:
        if np.random.rand() < threshold:
            features = trajectory_split(line)
            dt_features['Trajectory'].append(features[0])
            dt_features['HOG'].append(features[1])
            dt_features['HOF'].append(features[2])
            dt_features['MBHx'].append(features[3])
            dt_features['MBHy'].append(features[4])
        line = t.next()

#clustering
model_dict = {}
n_clusters = 10 
for feature in dt_features:
	X = np.array(dt_features[feature])
	print 'clustering %s features' % feature + ' of dimension: ',X.shape 
	model_dict[feature] = KMeans(n_clusters=n_clusters).fit(X)
	print 'done'

#save the models
save_dir = './codebook' #where to save the codebooks
if not os.path.isdir(save_dir):
	os.mkdir(save_dir)
joblib.dump(model_dict, save_dir + '/model_dict.pkl')
