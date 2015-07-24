#functions to call DenseTrack binary through python. line_to_dict
#converts the DenseTrack output to python lists line-by-line
import re
import subprocess

#set default parameters
dt_bin = '../release/DenseTrack' #location of the DenseTrack binary
dtp = ({'stride': 10,
        'traj_length': 15,
        'neigh_size': 32,
        'spatial_cells': 2,
        'time_cells': 3
        }) #dense trajectory parameters

def trajectory_generator(video, dt_bin=dt_bin, dtp=dtp):

    #convert dict to feed to terminal commant
    dtp_terminal = (['-W', str(dtp['stride']),
                     '-L', str(dtp['traj_length']),
                     '-N', str(dtp['neigh_size']),
                     '-s', str(dtp['spatial_cells']),
                     '-t', str(dtp['time_cells'])
                     ]) 

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
    trajectory_feature = line_vec[dtfi['Trajectory']:dtfi['HOG']]
    hog_feature = line_vec[dtfi['HOG']:dtfi['HOF']]
    hof_feature = line_vec[dtfi['HOF']:dtfi['MBHx']]
    mbhx_feature = line_vec[dtfi['MBHx']:dtfi['MBHy']]
    mbhy_feature = line_vec[dtfi['MBHy']:]
    return trajectory_feature, hog_feature, hof_feature, mbhx_feature, mbhy_feature
