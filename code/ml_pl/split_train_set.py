import os, math
def chunks(l,n):
	for i in xrange(0, len(l), n):
		yield l[i:i+n]

dataset_path = './raw_data'
train_files = './traintest_split/train_files_noduds.txt'
train_videos = []
with open(train_files, 'rb') as f:
	for video_name in f:
		train_videos.append(video_name[:-1])

#split into 3 separate lists
ch = chunks(train_videos, int(math.ceil(len(train_videos)/float(3))))

#write into new files 
for ii, c in enumerate(ch):
	#construct filename
	fn = './traintest_split/train_files_noduds_chunk' + str(ii) + '.txt'
	with open(fn, 'wb') as f:
		for st in c:
			f.write(st + '\n')
