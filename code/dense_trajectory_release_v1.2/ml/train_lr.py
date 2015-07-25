#train a regressor
import numpy as np
import scipy.sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from extract_dt import trajectory_generator, trajectory_split, FeatureDict

def video_descibe(video, model_dict):
    #encode a video into dense trajectory BoF features (using model_dict codebook learned previously)
    ohe = OneHotEncoder(n_values=model_dict['HOG'].get_params()['n_clusters']) #all models have same n_clusters
    t = trajectory_generator(video) #instantiate the trajectory generator
    trajectory = t.next() #generate the first trajectory
    #initialise the ohe trajectory features structure 
    dt_bof = FeatureDict(sparse=True)
    while trajectory:
        
        #get raw features of video
        features  = trajectory_split(trajectory)

        #loop through features, assigning cluster number
        for feature in features:
            #feature_cluster.append(model_dict[feature].predict(features[feature])[0])
            feature_cluster = model_dict[feature].predict(features[feature])[0]
            bof_rep = ohe.fit_transform(feature_cluster)
            dt_bof.add_element(feature=feature, element=bof_rep)#add to list
            
        #generate next trajectory
        trajectory = t.next()

    #take the sum across trajectories as the video representation
    dt_bof.calc_sums(normalise=True)
    return dt_bof.feature_sums

###
#generate the dataset for input to sklearn learners
###

#load codebook model
save_dir = './codebook'
model_dict = joblib.load(save_dir + '/model_dict.pkl')
#video list
videos = ['../test_sequences/low_res.ogv', '../test_sequences/low_res2.ogv']
dt_dataset = FeatureDict() #initialise structure (here each list is a video)
for ii,video in enumerate(videos):
    print 'processing video %i of %i' % (ii+1, len(videos))
    descriptors = video_descibe(video, model_dict)
    for feature in descriptors:
        dt_dataset.add_element(feature, descriptors[feature])
    



