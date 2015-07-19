#same as sOF but initiate sparse OF with facial points
import numpy as np
import cv2
import time

#cap = cv2.VideoCapture("walking_ex1.avi")
cap = cv2.VideoCapture(0)
time.sleep(1)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10,
					   qualityLevel = 0.1,
					   minDistance = 3,
					   blockSize = 5 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
				  maxLevel = 2,
				  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find points to track
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#detect face
old_gray = cv2.equalizeHist(old_gray)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faces = face_cascade.detectMultiScale(old_gray, scaleFactor=1.3, minNeighbors=4, minSize=(20,20), flags=cv2.cv.CV_HAAR_SCALE_IMAGE )
if len(faces) == 0:
	print('no faces detected')

#find points to track within faces (centred on eyes)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
for x,y,w,h in faces:

	#get   face roi
	roi_gray = old_gray[y:y+h, x:x+w]
	#find eyes within the roi
	eyes = eye_cascade.detectMultiScale(roi_gray)

	#find good points within the eyes
	p0_eye_list=[]#list of eye points

	for ex,ey,ew,eh in eyes:

		roi_eyes = roi_gray[ey:ey+eh, ex:ex+ew]
		eye_points = cv2.goodFeaturesToTrack(roi_eyes, mask = None, **feature_params)

		#convert coords to full image
		for i in range(len(eye_points)):
			eye_points[i][0][0] = eye_points[i][0][0] + x + ex
			eye_points[i][0][1] = eye_points[i][0][1] + y + ey

		#append the list of tracking points
		p0_eye_list.append(eye_points)


p0 = np.vstack((p0_eye_list[0], p0_eye_list[1]))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
	ret,frame = cap.read()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

	# Select good points
	good_new = p1[st==1]

	# draw the tracked points
	for t_point in good_new:
		cv2.circle(frame, (t_point[0], t_point[1]), 5, (0,255,0), -1)

	cv2.imshow('frame',frame)

	#waitkey
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

	# Now update the previous frame and previous points
	old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()