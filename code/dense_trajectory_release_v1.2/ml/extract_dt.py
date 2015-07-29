from __future__ import division
#functions to call DenseTrack binary through python. 
#converts the DenseTrack output to  a python dict of lists, separated by feature type
import re
import subprocess
from scipy.sparse import vstack
import scipy.sparse

#set default parameters
dt_bin = '../release/DenseTrack' #location of the DenseTrack binary
dtp = ({'stride': 10,
        'traj_length': 15,
        'neigh_size': 32,
        'spatial_cells': 2,
        'time_cells': 3
        }) #dense trajectory parameters
#convert dict to feed to terminal commant
dtp_terminal = (['-W', str(dtp['stride']),
                 '-L', str(dtp['traj_length']),
                 '-N', str(dtp['neigh_size']),
                 '-s', str(dtp['spatial_cells']),
                 '-t', str(dtp['time_cells'])
                 ]) 

#generator function for reading terminal output
def trajectory_generator(video, dt_bin=dt_bin, dtp_terminal=dtp_terminal):
    proc = subprocess.Popen([dt_bin, video] + dtp_terminal, stdout=subprocess.PIPE)
    while True:
        trajectory = proc.stdout.readline()
        if trajectory != '':
            yield trajectory
        else:
            yield None

def trajectory_split(trajectory, dtp=dtp):

	#calculate the length of each feature in this order: [metadata, trajectory, hog, hof, mbhx, mbhy]
	cell_den = dtp['spatial_cells']**2 * dtp['time_cells']
    dtfl =  [10, 2*dtp['traj_length'], 8*cell_den, 9*cell_den, 8*cell_den, 8*cell_den]            
    #get the index in the line for each feature (in same order)
	dtfi = [sum(dtfl[0:x]) for x in range(len(dtfl))]

    #vectorise the trajectory
    line_vec = re.split(r'\t', trajectory)[:-1] #remove the /n newline
    line_vec = [float(x) for x in line_vec] #turn chars into floats

    #split line into feature types
    t_dict = {}
	features = ['Trajectory', 'HOG', 'HOF','MBHx', 'MBHy']
	for ii,feature in enumerate(features, start=1): #start from 1 as excluding metadata
		t_dict[feature] = line_vec[dtfi[ii]:dtfi[ii]+dtfl[ii]]

    return t_dict 

#simple class for holding and adding dt features
class FeatureDict(object):
    def __init__(self, feature_list = ['Trajectory', 'HOG', 'HOF','MBHx', 'MBHy'], sparse=False):
        self.sparse = sparse #to handle OHE
        self.feature_dict = {}
        self.feature_sums = {}
        for key in feature_list:
            self.feature_dict.update({key:[]})
            self.feature_sums.update({key:[]})

    def add_element(self, feature, element):
        #append element (list or csr_matrix) to the list of lists (or csr_matrices)
        assert isinstance(element, (list, scipy.sparse.csr_matrix))
        if self.feature_dict[feature]:
            if not self.sparse:
                assert len(element) == len(self.feature_dict[feature][-1])
            else:
                assert element.shape[1] == self.feature_dict[feature][-1].shape[1]
        self.feature_dict[feature].append(element)

    def add_trajectory(self, trajectory, dtp=dtp):
        #add full trajectory from line
        features = trajectory_split(trajectory, dtp=dtp)
        assert len(features) == len(self.feature_dict)
        for feature in self.feature_dict:
            #self.add_element(feature, features[feature])
            self.feature_dict[feature].append(features[feature])
