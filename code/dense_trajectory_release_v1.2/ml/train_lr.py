#train a regressor
import numpy as np
import scipy.sparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from extract_dt import trajectory_generator, trajectory_split 

#load codebook model
save_dir = './codebook'
model_dict = joblib.load(save_dir + '/model_dict.pkl')

#video list
videos = ['../test_sequences/low_res.ogv', '../test_sequences/low_res2.ogv']

#describe each video with BoF descriptor
video = videos[0]
ohe = OneHotEncoder(n_values=model_dict['HOG'].get_params()['n_clusters'])
t = trajectory_generator(video) #instantiate the trajectory generator
trajectory = t.next() #generate the first trajectory
sparse_mat_list = []
olist = []
while trajectory:
    
    #get raw features of video
    features = {}
    features['Trajectory'], features['HOG'], features['HOF'], features['MBHx'], features['MBHy']  = trajectory_split(trajectory)

    #initialise list of clusters (list because multiple features) 
    feature_cluster = []
    #loop through features, assigning cluster number
    for feature in features:
        feature_cluster.append(model_dict[feature].predict(features[feature])[0])

    #one hot encode the feature clusters 
    trajectory_bof = ohe.fit_transform(feature_cluster)
    sparse_mat_list.append(trajectory_bof)#add to list

    #generate next trajectory
    trajectory = t.next()

#take the sum across trajectories as the video representation
video_bof_csr = scipy.sparse.vstack(sparse_mat_list).sum(axis=0)        
video_bof_arr = np.array(video_bof_csr)[0] #transform into vector
