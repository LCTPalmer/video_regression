from __future__ import division
#functions to call DenseTrack binary through python. line_to_dict
#converts the DenseTrack output to python lists line-by-line
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

def trajectory_generator(video, dt_bin=dt_bin, dtp_terminal=dtp_terminal):
    proc = subprocess.Popen([dt_bin, video] + dtp_terminal, stdout=subprocess.PIPE)
    while True:
        trajectory = proc.stdout.readline()
        if trajectory != '':
            yield trajectory
        else:
            yield None

def trajectory_split(trajectory, dtp=dtp):

    #calculate the length of each feature
    dtfl =               ({'Metadata': 10,
                           'Trajectory': 2 * dtp['traj_length'],
                           'HOG': 8 * dtp['spatial_cells']**2 * dtp['time_cells'],
                           'HOF': 9 * dtp['spatial_cells']**2 * dtp['time_cells'],
                           'MBHx': 8 * dtp['spatial_cells']**2 * dtp['time_cells'],
                           'MBHy': 8 * dtp['spatial_cells']**2 * dtp['time_cells'],
                           })
    #get the index in the line for each feature
    dtfi =               ({'Metadata': 0,
                           'Trajectory': dtfl['Metadata'],
                           'HOG': dtfl['Metadata']+dtfl['Trajectory'],
                           'HOF': dtfl['Metadata']+dtfl['Trajectory']+dtfl['HOG'],
                           'MBHx': dtfl['Metadata']+dtfl['Trajectory']+dtfl['HOG']+dtfl['HOF'],
                           'MBHy': dtfl['Metadata']+dtfl['Trajectory']+dtfl['HOG']+dtfl['HOF']+dtfl['MBHx'],
                           })

    #vectorise
    line_vec = re.split(r'\t', trajectory)[:-1] #remove the /n newline
    line_vec = [float(x) for x in line_vec] #turn chars into floats
    #split line into feature types
    t_dict = {}
    t_dict['Trajectory'] = line_vec[dtfi['Trajectory']:dtfi['HOG']]
    t_dict['HOG'] = line_vec[dtfi['HOG']:dtfi['HOF']]
    t_dict['HOF'] = line_vec[dtfi['HOF']:dtfi['MBHx']]
    t_dict['MBHx'] = line_vec[dtfi['MBHx']:dtfi['MBHy']]
    t_dict['MBHy'] = line_vec[dtfi['MBHy']:]
    return t_dict 

#class for holding and manipulating dt features
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

    def calc_sums(self, normalise=False):
        #calculate the sum of lists/csr_matrices 
        for feature in self.feature_dict:
            if not self.sparse:
                self.feature_sums[feature] = [sum(x) for x in zip(*self.feature_dict[feature])]
            else:
                self.feature_sums[feature] = vstack(self.feature_dict[feature]).sum(axis=0).tolist()[0]
            if normalise:
                s = sum(self.feature_sums[feature])
                self.feature_sums[feature] = [x/s for x in self.feature_sums[feature]]
