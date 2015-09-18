"""
Python code for loading Hollywood2 meta-data into Python objects.
The dataset is a pair of video classification problems: Actions and Scenes.
The number of rows varies from 224 to 576.
The number of columns varies from 480 to 720.
The aspect ratio is also not fixed, it varies from 1.25 to 2.5.
"""
import sys, os
#import pyffmpeg
action_names = ['AnswerPhone', 'DriveCar', 'Eat',
'FightPerson' , 'GetOutCar', 'HandShake',
'HugPerson', 'Kiss', 'Run', 'SitDown',
'SitUp', 'StandUp']
def load_basic_info(avidir, vid_info=True):
	"""
	Return a dictionary of dictionaries.
	Each [internal] dictionary corresponds to a clip from the avidir.
	The keys in each [internal] dictionary are:
	name: the name of the file in avidir (without extension)
	path: the full path of the file in avidir
	label: a string describing the subject of the video.
	For actions this is one of the `action_names`.
	For scenes this is one of the `scene_names`.
	group: one of 'train', 'test', 'autotrain'
	resolution: (height, width)
	duration: number of frames
	shots: first frames of shots within the clip
	The outer dictionary maps name -> internal dictionary
	"""
	clips = {}
	for filename in os.listdir(avidir):
		# initialize the list of clips with just name and path
		name = filename[:-4]
		clip = dict(
		path=os.path.join(avidir,filename),
		name=name)
		print clip['name']
		# set the group based on the filename
		if 'cliptrain' in filename:
			clip['group'] = 'train'
		elif 'cliptest' in filename:
			clip['group'] = 'test'
		elif 'clipauto' in filename:
			clip['group'] = 'autotrain'
		else:
			assert False
		clips[name] = clip
	return clips

def add_label_info(clips, labeldir, labels, group_names):
	for label in labels:
		for group in group_names:
			filename = '%s_%s.txt' % (label, group)
			for line in open(os.path.join(labeldir, filename)):
				name, one_or_neg1 = line.split()
				clips[name].setdefault('label', {})[label] = (one_or_neg1 == '1')

	# assert that every clip got a full set of labels
	for name, clip in clips.iteritems():
		assert len(clip['label']) == len(labels)

def add_shots_info(clips, shotdir):
	for name, clip in clips.iteritems():
		clip['shots'] = [int(f)
			for f in open(os.path.join(shotdir, name+'.sht')).read().split()]
		print clip['shots']

def build(root):
	"""
	Return a pair of lists.
	The first is a list of movies in the Actions set
	The second is a list of movies in the Scenes set
	"""
	actions = load_basic_info( os.path.join(root, 'AVIClips',) )
	add_label_info(actions, os.path.join(root, 'ClipSets',),
	action_names, ['train', 'test', 'autotrain'])
	add_shots_info(actions, os.path.join(root, 'ShotBounds',))

	return actions

#create the python object dataset version
hollywood2_dir = '~/projects/Downloads/hollywood2'
save_loc = './dataser/hollywood2.pkl'

import os
import cPickle as pickle
if not os.path.isfile(save_loc): #make a saved python object version
	actions = build(hollywood2_dir)
	f = open(save_loc, 'wb')
	pickle.dump(actions, f)
	f.close()
else: #load up that version
    f = open(save_loc, 'rb')
	actions = pickle.load(f)
	f.close()	

for clip in actions:
    l = []
    for k,v in a[clip]['label'].iteritems():
        if v:
            l.append(k)
            print clip, l, a[clip]['shots']

