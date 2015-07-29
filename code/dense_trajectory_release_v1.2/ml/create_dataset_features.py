#train a regressor
import os
import numpy as np
import scipy.sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from extract_dt import trajectory_generator, FeatureDict
from load_hollywood import load_data

def video_describe(video, model_dict, dtp):
	#encode a video into dense trajectory BoF features (using model_dict codebook learned previously)
	t = trajectory_generator(video=video, dtp=dtp) #instantiate the trajectory generator

	#extract raw trajectories for video
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

###
#generate and save the dataset for input to sklearn learners
###

#load codebook model
codebook_dir = './codebook'
model_dict = joblib.load(codebook_dir + '/model_dict.pkl')
#video file list
hollywood_dir = '/home/lpa8529/projects/Downloads/hollywood' #directory of the dataset
dataset_type = 'test' #training set or test set ('train' or 'test')
videos, frame_ranges, labels = load_data(hollywood_dir, dataset_type)
dt_dataset = FeatureDict(feature_list = ['Trajectory', 'HOG', 'HOF', 'MBHx', 'MBHy', 'Label']) #initialise structure with Label field
reject_list = [] #list of clips not processed

#loop through videos adding descriptors to dt structure
for ii,(video, frame_range, label) in enumerate(zip(videos, frame_ranges, labels), start=1):

	print 'processing video %i of %i' % (ii, len(videos))

	#set the dt parameters (same as used for learning the codebook)
	clip_length = int(frame_range[1])-int(frame_range[0])
	L = str(min([clip_length, 15])) #minimum length as length of trajectories
	if clip_length<15: print 'video %i: %s is less than 15 frames long (%i frames)' % (ii,video,clip_length)
	dtp = {'-S': frame_range[0], '-E': frame_range[1], '-W': 10, '-L': L}#start and end frames

	#generate the video descriptors
	descriptors = video_describe(video=video, model_dict=model_dict, dtp=dtp)

	#add the descriptors for each feature to the dataset struct
	for feature in descriptors:
		dt_dataset.add_element(feature, descriptors[feature])

	#add class for this video
	dt_dataset.add_element('Label', [label]) #keep same format as other features (list)

#save dataset
dataset_dir = './dataset'
if not os.path.isdir(dataset_dir):
	os.mkdir(dataset_dir)
joblib.dump(dt_dataset, dataset_dir + '/' + dataset_type + '.pkl')
