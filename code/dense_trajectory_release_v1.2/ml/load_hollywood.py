#load hollywood action dataset
#return list of file locations of videos and associated labels
import re

def load_data(hollywood_dir, dataset_type):
    #load list of clip locations and associated action
    video_root = hollywood_dir + '/videoclips/'
    anno_file = hollywood_dir + '/annotations/' + dataset_type + '_clean.txt'

    #read through the annotation file
    video_list = []
    class_list = []
    with open(anno_file, 'rb') as f:
        line = f.readline()
        while line:
            current_video = re.search('"(.*)"', line).groups()[0]
            current_class = re.search('<(.*)>', line).groups()[0]
            video_list.append(video_root + current_video)
            class_list.append(current_class)
            line = f.readline()
    return video_list, class_list
