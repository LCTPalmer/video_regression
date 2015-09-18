import os
import numpy as np
import scipy.sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from extract_dt import trajectory_generator, trajectory_split, FeatureDict

###--- LEARNING THE CODEBOOK --###

def sample_trajectories(videos, ratio=.002, save_dir='./codebook', samples_save_name='pl_samples.pkl', load_precalc=True):

	if load_precalc:
		if os.path.isfile(os.path.join(save_dir, samples_save_name)):
			#load pre calculated samples
			return joblib.load(os.path.join(save_dir, samples_save_name))
		else:
			raise ValueError, 'No precalculated samples with given filename - check filename or set load_precalc=False'

	else:		
		#instantiate trajectory features object
		dtf = FeatureDict()
		for ii, video in enumerate(videos, start=1):
			print 'processing video {0} of {1}'.format(ii, len(videos))
			t = trajectory_generator(video=video) #instantiate generator
			trajectory = t.next()
			t_cnt, samp_cnt = 0, 0
			while trajectory:
				if np.random.rand() < ratio:
					dtf.add_trajectory(trajectory)
					samp_cnt += 1
				t_cnt += 1
				trajectory = t.next()
			print 'contained {0} total trajectories, {1} sampled'.format(t_cnt, samp_cnt)
		#save samples
		joblib.dump(dtf, os.path.join(save_dir, samples_save_name))

		#return the sampled trajectories
		return dtf

def cluster_trajectories(dtf, n_clusters=4000, save_dir='./codebook', model_save_name='model_dict.pkl', load_precalc=True):

	if load_precalc:
		if os.path.isfile(os.path.join(save_dir, model_save_name)):
			return joblib.load(os.path.join(save_dir, model_save_name))
		else:
			raise ValueError, 'No precalculated model with given filename - check filename or set load_precalc=False'

	else:
		model_dict = {}
		for feature in dtf.feature_dict:
			X = np.array(dtf.feature_dict[feature])
			print 'clustering {0} features of dimension: {1}'.format(feature, X.shape)
			model_dict[feature] = MiniBatchKMeans(n_clusters=n_clusters, verbose=True, reassignment_ratio=0, max_no_improvement=1000).fit(X)
			print 'done'
		#save the models
		joblib.dump(model_dict, os.path.join(save_dir, model_save_name))

		#return the model
		return model_dict


###--- EXTRACTING THE DATASET USING A CODEBOOK ---###

def feature_extract(videos, model_dict, labels=False, save_dir='./dataset', dataset_name='train_set.pkl', load_precalc=True):

	#define function for one video 
	def video_describe(video, model_dict):
		t = trajectory_generator(video=video) #instantiate the trajectory generator
		trajectory = t.next() #generate the first trajectory
		dt_vid = FeatureDict() #structure for video trajectories
		while trajectory:
			dt_vid.add_trajectory(trajectory)
			trajectory = t.next()

		dt_bof = FeatureDict() #structure for OHE features output
		#loop through features, assigning cluster number
		for feature in dt_vid.feature_dict:
			#assign cluster value to each trajectory
			cur = np.array(dt_vid.feature_dict[feature])#turn into numpy array for OHE and KMeans
			feature_cluster = np.reshape(model_dict[feature].predict(cur), (cur.shape[0], 1))#reshape - keep rows as rows for OHE
			#extract OHE features
			ohe = OneHotEncoder(n_values=model_dict[feature].get_params()['n_clusters']) #instantiate a OHE model
			bof_sum = ohe.fit_transform(feature_cluster).sum(axis=0).tolist()[0]
			dt_bof.feature_dict[feature] = bof_sum 

		return dt_bof.feature_dict

	if load_precalc:
		if os.path.isfile(os.path.join(save_dir, dataset_type)):
			return joblib.load(os.path.join(save_dir, dataset_type))
		else:
			raise ValueError, 'No precalculated dataset with given filename - check filename or set load_precalc=False'
	
	else:
		#initialise the extracted features directory
		dt_dataset = FeatureDict(feature_list = ['Trajectory', 'HOG', 'HOF', 'MBHx', 'MBHy', 'Video_Name', 'Label'])

		#loop through videos adding descriptors to dt structure
		for ii, video in enumerate(videos, start=1):
			print 'extracting features from video {0} of {1}'.format(ii, len(videos))
			#generate the video descriptors
			descriptors = video_describe(video=video, model_dict=model_dict)
			#add the descriptors for each feature to the dataset struct
			for feature in descriptors:
				dt_dataset.add_element(feature, descriptors[feature])
			#add video name
			dt_dataset.add_element('Video_Name', [os.path.basename(video)])
			#add class for this video
			if labels:
				dt_dataset.add_element('Label', [label]) #keep same format as other features (list)
			else:
				dt_dataset.add_element('Label', [None])
		#save the features
		joblib.dump(dt_dataset, os.path.join(save_dir, dataset_name))

		#return the dataset
		return dt_dataset
