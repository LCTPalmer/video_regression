#example of the improved_trajectories feature extraction pipeline
import os
from sklearn.externals import joblib
from bof_feature_extraction import sample_trajectories, cluster_trajectories, feature_extract

#setup
pipeline = 		{
			'sample_trajectories': {'load_precalc': False}, #have trajectories already been sampled and would you like to use them?
			'learn_codebook': {'load_precalc': False} #has the codebook model already been learned and would you like to use it?
			}

#-------------------------------------------------------------------------------------------------------

###--- PARAMS ---###

#DATASET
videos_path = './raw_data' #directory containing the raw videos
train_files = './traintest_split/train_files.txt' #file containing list of video names in the training set
test_files = './traintest_split/test_files.txt' #as above but for the test set

#create the full lists of videos
train_videos = []
test_videos = []
with open(train_files, 'rb') as f:
	for video_name in f:
		train_videos.append(os.path.join(videos_path, video_name[:-1])) #-1 to remove the newline character
with open(test_files, 'rb') as f:
	for video_name in f:
		test_videos.append(os.path.join(videos_path, video_name[:-1]))

#SAMPLING TRAJECTORIES
ratio = .0018 #what proportion of the trajectories in the whole training set to use to learn the codebook
sample_save_dir = './codebook' #where to save the samples
if not os.path.isdir(sample_save_dir):
	os.mkdir(sample_save_dir)

#CLUSTERING
n_clusters = 4000
model_save_dir = './codebook' #where to save the codebooks
if not os.path.isdir(model_save_dir):
	os.mkdir(model_save_dir)

#FEATURE EXTRACTION
dataset_types = {'train': train_videos, 'test': test_videos} #which sets to extract features for
dataset_save_dir = './dataset'
if not os.path.isdir(dataset_save_dir):
	os.mkdir(dataset_save_dir)

#-------------------------------------------------------------------------------------------------------

###--- RUN THE PIPELINE ---###

#--------------------

#sample
pc = pipeline['sample_trajectories']['load_precalc']
samples = sample_trajectories(train_videos, ratio=ratio, save_dir=sample_save_dir, samples_save_name='pl_samples.pkl', load_precalc=pc)

#--------------------

#cluster
pc = pipeline['learn_codebook']['load_precalc']	
model_dict = cluster_trajectories(samples, n_clusters=n_clusters, 
	save_dir=model_save_dir, model_save_name='model_dict.pkl', load_precalc=pc)

#---------------------

full_dataset = {}
for dataset_type in dataset_types:
	dataset_name = dataset_type + '_set.pkl'
	full_dataset[dataset_type] = feature_extract(dataset_types[dataset_type], model_dict, labels=False, 
		save_dir=dataset_save_dir, dataset_name=dataset_name, load_precalc=pc)

#-------------------------------------------------------------------------------------------------------



