#run the improved_trajectories featue extraction pipeline
import os
from sklearn.externals import joblib
from bof_feature_extraction import sample_trajectories, cluster_trajectories, feature_extract

#setup
pipeline = {
			'sample_trajectories': {'return': True, 'load_precalc': True}, 
			'learn_codebook': {'return': True, 'load_precalc': True},
			'extract_features': {'return': True, 'load_precalc': False},
			}

#--------------------------------------------------------------------------
###--- PARAMS ---###
###dataset###
dataset_path = './raw_data' #directory containing the videos
train_files = './traintest_split/train_files_noduds_chunk0.txt'
test_files = './traintest_split/test_files_noduds.txt'

#create the full lists of videos
train_videos = []
test_videos = []
with open(train_files, 'rb') as f:
	for video_name in f:
		train_videos.append(os.path.join(dataset_path, video_name[:-1]))
with open(test_files, 'rb') as f:
	for video_name in f:
		test_videos.append(os.path.join(dataset_path, video_name[:-1]))

# ##############################FORTESTING
# train_videos = train_videos[:1]
# test_videos = test_videos[:2]
# ###############################

###clustering###
n_clusters = 4000
model_save_dir = './codebook' #where to save the codebooks
sample_save_dir = './codebook'
if not os.path.isdir(model_save_dir):
	os.mkdir(model_save_dir)
if not os.path.isdir(sample_save_dir):
	os.mkdir(sample_save_dir)

###feature extract###
dataset_types = {'train_chunk0': train_videos} #which sets to extract features for
dataset_save_dir = './dataset'
if not os.path.isdir(dataset_save_dir):
	os.mkdir(dataset_save_dir)

#----------------------------------------------------------------------------
###--- RUN THE PIPELINE ---###

if pipeline['sample_trajectories']['return']:
	pc = pipeline['sample_trajectories']['load_precalc']

	#sample
	samples = sample_trajectories(train_videos, ratio=.0018, load_precalc=pc)

#--------------------

if pipeline['learn_codebook']['return']:
	pc = pipeline['learn_codebook']['load_precalc']
	
	#cluster
	model_dict = cluster_trajectories(samples, n_clusters=n_clusters, 
		save_dir=model_save_dir, model_save_name='model_dict.pkl', load_precalc=pc)

#---------------------

if pipeline['extract_features']['return']:
	pc = pipeline['extract_features']['load_precalc']
	full_dataset = {}
	for dataset_type in dataset_types:
		dataset_name = dataset_type + '_set.pkl'
		full_dataset[dataset_type] = feature_extract(dataset_types[dataset_type], model_dict, labels=False, 
			save_dir=dataset_save_dir, dataset_name=dataset_name, load_precalc=pc)

#---------------------------------------------------------------------------



