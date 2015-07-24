import subprocess #running dense_trajectories
import re
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib

#args
dt_bin = '../release/DenseTrack' #location of the DenseTrack binary
videos = ['../test_sequences/low_res.ogv', '../test_sequences/low_res2.ogv'] #video file list
save_dir = './codebook' #where to save the codebooks
dtp =       ({'stride': 10,
			  'traj_length': 15,
			  'neigh_size': 32,
			  'spatial_cells': 2,
			  'time_cells': 3
			  })

#convert dict to feed to terminal commant
dtp_terminal =       (['-W', str(dtp['stride']),
					   '-L', str(dtp['traj_length']),
					   '-N', str(dtp['neigh_size']),
					   '-s', str(dtp['spatial_cells']),
					   '-t', str(dtp['time_cells'])
		               ]) 
#calculate the number of features for each type
dtfl =               ({'Metadata': 10,
					   'Trajectory': 2 * dtp['traj_length'],
					   'HOG': 8 * dtp['spatial_cells']**2 * dtp['time_cells'],
					   'HOF': 9 * dtp['spatial_cells']**2 * dtp['time_cells'],
					   'MBHx': 8 * dtp['spatial_cells']**2 * dtp['time_cells'],
					   'MBHy': 8 * dtp['spatial_cells']**2 * dtp['time_cells'],
					   })
#get the index in the line for each type
dtfi =               ({'Metadata': 0,
					   'Trajectory': dtfl['Metadata'],
					   'HOG': dtfl['Metadata']+dtfl['Trajectory'],
					   'HOF': dtfl['Metadata']+dtfl['Trajectory']+dtfl['HOG'],
					   'MBHx': dtfl['Metadata']+dtfl['Trajectory']+dtfl['HOG']+dtfl['HOF'],
					   'MBHy': dtfl['Metadata']+dtfl['Trajectory']+dtfl['HOG']+dtfl['HOF']+dtfl['MBHx'],
					   })

#sampling strategy
#sample only approximately for now...
	#assumptions: num videos = 2000
	#			  num trajectories / video = 2000
	#			  total trajectories wanted = 100000
	#solution: take 50 trajectories from each video
	#		   prob per trajectory of 100,000/4,000,000 = 0.025
threshold = 0.025

#initialise trajectory features structure 
dt_features = ({'Trajectory': [],
					  'HOG': [],
					  'HOF': [],
					  'MBHx': [],
					  'MBHy': []
					  })

#read in features one video at a time
for ii, video in enumerate(videos):
	print 'processing video: %s (%i of %i)' % (video,ii+1,len(videos))
	proc = subprocess.Popen([dt_bin, video] + dtp_terminal, stdout=subprocess.PIPE)
	while True:
		#read stdout lines into dt_features structure
		line = proc.stdout.readline()
		if np.random.rand()<threshold: #simple sampling based on threshold defined above
			if line != '':
				#vectorise
				line_vec = re.split(r'\t', line)[:-1] #remove the /n newline
				line_vec = [float(x) for x in line_vec] #turn chars into floats
				#split line into feature types
				dt_features['Trajectory'].append(line_vec[dtfi['Trajectory']:dtfi['HOG']])
				dt_features['HOG'].append(line_vec[dtfi['HOG']:dtfi['HOF']])
				dt_features['HOF'].append(line_vec[dtfi['HOF']:dtfi['MBHx']])
				dt_features['MBHx'].append(line_vec[dtfi['MBHx']:dtfi['MBHy']])
				dt_features['MBHy'].append(line_vec[dtfi['MBHy']:])
			else:
				break

#clustering
model_dict = {}
n_clusters = 10 
for feature in dt_features:
	X = np.array(dt_features[feature])
	print 'clustering %s features' % feature + ' of dimension: ',X.shape 
	model_dict[feature] = KMeans(n_clusters=n_clusters).fit(X)
	print 'done'

#save the models
if not os.path.isdir(save_dir):
	os.mkdir(save_dir)
joblib.dump(model_dict, save_dir + '/model_dict.pkl')




