#import modules for the HOF project
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import sklearn
import scipy
import os
from skimage.feature import hog

def cv2imshow(im, name = ''):
	cv2.imshow(name, im)
	
	while True:
		k = cv2.waitKey()
		if k == 27:
			cv2.destroyAllWindows()
			break