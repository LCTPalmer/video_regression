#train a regressor
import numpy as np
import scipy.sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from extract_dt import trajectory_generator, trajectory_split, FeatureDict
from load_hollywood import load_data

def video_describe(video, model_dict, dtp_terminal):
	#encode a video into dense trajectory BoF features (using model_dict codebook learned previously)
	ohe = OneHotEncoder(n_values=model_dict['HOG'].get_params()['n_clusters']) #all models have same n_clusters
	t = trajectory_generator(video=video, dtp_terminal=dtp_terminal) #instantiate the trajectory generator
	trajectory = t.next() #generate the first trajectory
	#initialise the ohe trajectory features structure 
	dt_vid = FeatureDict()
	while trajectory:
		#add trajectory
		dt_vid.add_trajectory(trajectory)
		#generate next trajectory
		trajectory = t.next()

	dt_bof = FeatureDict()
	#loop through features, assigning cluster number
	for feature in dt_vid.feature_dict:
		cur = np.array(dt_vid.feature_dict[feature])#turn into numpy array for OHE and KMeans
		feature_cluster = np.reshape(model_dict[feature].predict(cur), (cur.shape[0], 1))#reshape - keep rows as rows for OHE
		dt_bof.feature_dict[feature] = ohe.fit_transform(feature_cluster)
		bof_sum = dt_bof.feature_dict[feature].sum(axis=0).tolist()[0]
		dt_bof.feature_sums[feature] = bof_sum 

	return dt_bof.feature_sums

###
#generate and save the dataset for input to sklearn learners
###

#load codebook model
codebook_dir = './codebook'
model_dict = joblib.load(codebook_dir + '/model_dict.pkl')
#video file list
hollywood_dir = '/home/lpa8529/projects/Downloads/hollywood' #directory of the dataset
dataset_type = 'train' #training set or test set
videos, frame_ranges, action_classes = load_data(hollywood_dir, dataset_type)
dt_dataset = FeatureDict(feature_list = ['Trajectory', 'HOG', 'HOF', 'MBHx', 'MBHy', 'ActionClass']) #initialise structure with class! (here each list is a video)

#loop through videos adding descriptors to dt structure
for ii,(video, frame_range, action_class) in enumerate(zip(videos, frame_ranges, action_classes), start=1):
	print 'processing video %i of %i' % (ii, len(videos))

	#set the dt parameters (same as used for learning the codebook)
	dtp = ['-S', frame_range[0], '-E', frame_range[1], '-W', '10']#start and end frames

	#generate the video descriptors
	descriptors = video_describe(video=video, model_dict=model_dict, dtp_terminal=dtp)

	#add the descriptors for each feature to the dataset struct
	for feature in descriptors:
		dt_dataset.add_element(feature, descriptors[feature])

	#add class for this video
	dt_dataset.add_element('ActionClass', [action_class]) #keep same format as other features (list)

#save dataset
dataset_dir = './dataset'
if not os.path.isdir(dataset_dir):
	os.mkdir(dataset_dir)
joblib.dump(dt_dataset, save_dir + '/' + dataset_type + '.pkl')
