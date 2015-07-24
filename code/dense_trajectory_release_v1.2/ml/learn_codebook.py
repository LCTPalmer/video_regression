import subprocess #running dense_trajectories
import re

#args
dt_bin = '../release/DenseTrack' #location of the DenseTrack binary
video = '../test_sequences/low_res2.ogv'

#sampling strategy
pass

#initialise output (list of trajectory features)
out_traj = []

#read in features one line at a time
proc = subprocess.Popen([dt_bin, video, '-W', '10'], stdout=subprocess.PIPE)
while True:
	line = proc.stdout.readline()
	if line != '':
		#vectorise
		line_vec = re.split(r'\t', line)[:-1] #remove the /n newline
		line_vec = [float(x) for x in line_vec] #turn chars into floats
		#print line_vec 
		out_traj.append(line_vec)
	else:
		break




