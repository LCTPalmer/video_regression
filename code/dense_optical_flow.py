import cv2
import numpy as np
import time

#cap = cv2.VideoCapture("walking_ex1.avi")
cap = cv2.VideoCapture('../data/walking_ex1.avi')
#cap = cv2.VideoCapture(0)
time.sleep(1)


#get first frame
scale_fac = 1
[ret, frame1] = cap.read()
frame1 = cv2.resize(frame1, (0,0), fx = scale_fac, fy = scale_fac)

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(ret):
    ret, frame2 = cap.read()
    frame2 = cv2.resize(frame2, (0,0), fx = scale_fac, fy = scale_fac)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 5, 3, 3, 1, False)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    # hsv[...,0] = ang*180/np.pi/2
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    # rgb_show = cv2.resize(rgb,(0,0),fx=4, fy=4)
    mag_show = cv2.resize(mag, (0,0), fx=4, fy=4)
    cv2.imshow('optical flow', mag_show)
    cv2.imshow('orig', frame2)

    k = cv2.waitKey(30)
    if k == 27:
        break

    prvs = next

cap.release()
cv2.destroyAllWindows()
