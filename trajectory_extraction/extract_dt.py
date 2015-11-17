from __future__ import division
#functions to call DenseTrack(Stab) binary through python. 
#converts the DenseTrack output to  a python dict of lists, separated by feature type
import re
import subprocess
from scipy.sparse import vstack
import scipy.sparse

#set default parameters
dt_bin = '../improved_trajectory_release/release/DenseTrackStab' #location of the DenseTrack binary
#dt_bin = '../dense_trajectory_release_v1.2/release/DenseTrack'
dtp = ({'-W': 5,  #stride
        '-L': 15, #trajectory length in frames
        '-N': 32, #neighbourhood size
        '-s': 2,  #spatial cell resolution
        '-t': 3   #temporal cell resolution
        })

def param_convert(dtp):
	'''convert param dict to terminal ready list of ['param1', 'value1', 'param2', 'value2', ...]'''
	param_list = []
	for param, value in dtp.items():
		param_list.append(param)
		param_list.append(str(value))
	return param_list

def trajectory_generator(video, dt_bin=dt_bin, dtp=dtp):
	'''generator function for reading terminal output'''
	dtp_terminal = param_convert(dtp)
	proc = subprocess.Popen([dt_bin, video] + dtp_terminal, stdout=subprocess.PIPE)
	while True:
		trajectory = proc.stdout.readline()
		if trajectory != '':
			yield trajectory
		else:
			yield None

def trajectory_split(trajectory, dtp=dtp):
	'''split the trajectory line (given by DenseTrack) into a dictionary of individual feature types'''
	#calculate the length of each feature in this order: [metadata, trajectory, hog, hof, mbhx, mbhy]
	cell_den = dtp['-s']**2 * dtp['-t']
	dtf_lengths =  [10, 2*dtp['-L'], 8*cell_den, 9*cell_den, 8*cell_den, 8*cell_den]            
	#get the index in the line for each feature (in same order)
	dtf_inds = [sum(dtf_lengths[0:x]) for x in range(len(dtf_lengths))]

	#vectorise the trajectory
	line_vec = re.split(r'\t', trajectory)[:-1] #remove the /n newline
	line_vec = [float(x) for x in line_vec] #turn chars into floats

	#split line into feature types
	feature_dict = {}
	features = ['Trajectory', 'HOG', 'HOF','MBHx', 'MBHy']
	for ii,feature in enumerate(features, start=1): #start from 1 as excluding metadata
		feature_dict[feature] = line_vec[dtf_inds[ii]:dtf_inds[ii]+dtf_lengths[ii]]

	return feature_dict 

class FeatureDict(object):
    '''simple class for holding and adding dt features'''
    def __init__(self, feature_list = ['Trajectory', 'HOG', 'HOF','MBHx', 'MBHy'], sparse=False):
        self.sparse = sparse #to handle OHE
        self.feature_dict = {}
        for key in feature_list:
            self.feature_dict.update({key:[]})

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
