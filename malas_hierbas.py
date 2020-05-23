# CENTER OF SHAPE APPLIET TO PLANT CENTER DETECTION
# import the necessary packages
import argparse
import imutils
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.feature import greycomatrix, greycoprops

# load the image, convert it to grayscale, blur it slightly, and threshold it
image = [cv2.imread(file) for file in glob.glob("Images/*.jpeg")]

i = int(sys.argv[1])
scale_percent = 40
width = int(image[i].shape[1]*scale_percent/100)
height = int(image[i].shape[0]*scale_percent/100)
dim = (width, height)
resized = cv2.resize(image[i], dim, interpolation = cv2.INTER_AREA)
blur = cv2.blur(resized, (5,5))

cv2.imshow('Resized', resized) 
cv2.waitKey(0)

hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_green = np.array([20,40,40])
upper_green = np.array([85,255,255])

thresh = cv2.inRange(hsv, lower_green, upper_green)
cv2.imshow('Thresh', thresh) 
cv2.waitKey(0)

# morphologycal operation kernel definition
"""kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
cv2.imshow('Opening', opening) 
cv2.waitKey(0)"""

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
sure_bg = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
cv2.imshow('Sure background', sure_bg) 
cv2.waitKey(0)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
ret, sure_fg = cv2.threshold(dist_transform,0.05*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow('Unknown', unknown) 
cv2.waitKey(0)

cv2.imshow('Sure foreground', sure_fg) 
cv2.waitKey(0)

# Marker labelling
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_bg)

# Add one to all labels so that sure background is not 0, but 1
labels = labels+1

# Now, mark the region of unknown with zero
labels[unknown==255] = 0
labels_ori = labels.copy()

area_thresh = 2000
greatest_areas = []
for i in range(1,num_labels):
	if stats[i, 4]>area_thresh:
		greatest_areas.append(i)
	else:
		labels[labels==i]=1

print(stats)
print(centroids)
print(greatest_areas)

labels_ori = cv2.watershed(resized,labels_ori)
resized[labels_ori == -1] = [255,0,0]


for i in range(len(greatest_areas)):
	obj = greatest_areas[i]
	#x = stats[obj,0]
	#y = stats[obj,1]
	j = stats[obj,2]
	k = stats[obj,3]
	fact = 0.2;
	
	print(int(centroids[obj,1]-k*fact))
	print(int(centroids[obj,1]+k*fact))
	


	crop = resized[int(centroids[obj,1]-k*fact):int(centroids[obj,1]+k*fact), int(centroids[obj,0]-j*fact):int(centroids[obj,0]+j*fact)]
	cv2.imshow('Cropped '+ str(i), crop)
	cv2.waitKey(0)

	# Draw centroids of selected objects
	resized = cv2.circle(resized, (int(centroids[obj,0]), int(centroids[obj,1])), 10, (255,0,0), 6)

cv2.imshow('Markers', resized) 
cv2.waitKey(0)

