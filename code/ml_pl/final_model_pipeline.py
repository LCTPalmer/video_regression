#trains a model on the training set, predicts on test set

#takes in a mapping from video_name to rating (location to this csv)
#as well as location to training and test features

import os
import numpy as np
from multichannel_svr.multichannel_svr import MultiChannelSVR

dataset_root = '/home/lukep86/projects/THE_dataset' #directory where features/labels kept
mapping_path = os.path.join(dataset_root, 'ratings_by_filename_it35.csv')
train_path = os.path.join(dataset_root, 'train_set.pkl')
test_path = os.path.join(dataset_root, 'test_set.pkl')

#function for turning feature_dict into multi-channel tuple
def feature_dict_to_tuple(feature_dict, feature_list=['Trajectory', 'HOG', 'HOF', 'MBHx', 'MBHy']):
	multichannel_list = []
	for feature in feature_list:
		multichannel_list.append(np.array(dataset[feature]))
	#make immutable
	return tuple(multichannel_list)

#function assigning labels to dataset
