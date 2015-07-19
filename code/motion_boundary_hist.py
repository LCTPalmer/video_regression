import cv2
import numpy as np
import time
from skimage.feature import hog

#inputs
video_address = '../data/walking_ex1.avi'
#spatial scaling factor
scale_fac = (.5, .5)
#temporal resolution - nth frame
temp_res = 5
#hog params
hp = dict(orientations=6, pixels_per_cell=(16,16), cells_per_block=(1,1), visualise=False)
#optical flow params
ofp = dict(orientations=6, pixels_per_cell=(16,16), cells_per_block=(1,1), visualise=False)


#the function:

#open the video capture stream
cap = cv2.VideoCapture(video_address)

MBH = []#initialise list for building MBH

#get first and second frames
[ret, frame1] = cap.read()
[ret, frame2] = cap.read()

#loop through frames - check that a frame has been returned by .read
while(ret):

    #resize and convert the frames to grayscale
    frame1 = cv2.resize(frame1, (0,0), fx = scale_fac[0], fy = scale_fac[1])
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2 = cv2.resize(frame2, (0,0), fx = scale_fac[0], fy = scale_fac[1])
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    #calculate dense optical flow betweek frames
    flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 5, 3, 3, 1, False)

    #calculate HOG of flow images (x and y separately) to give MBH
    hog_x = hog(flow[...,0], **hp)
    hog_y = hog(flow[...,1], **hp)

    #keep x and y 
    mbh_cur = np.hstack((hog_x, hog_y))
    MBH.append(mbh_cur)

    #show the magnitude hof image
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mag_show = cv2.resize(mag, (0,0), fx=4, fy=4)
    cv2.imshow('optical flow', mag_show)
    #see if exit button is held down
    k = cv2.waitKey(30)
    if k == 27:
        break

    #read next frame
    for it in range(temp_res):
        frame1 = frame2#keep old frame for flow calc
        ret, frame2 = cap.read()


cap.release()
cv2.destroyAllWindows()

MBH = np.hstack(MBH);