# CENTER OF SHAPE APPLIET TO PLANT CENTER DETECTION
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.feature import greycomatrix, greycoprops

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
	frame = vs.read()
	scale_percent = 70
	width = int(frame.shape[1]*scale_percent/100)
	height = int(frame.shape[0]*scale_percent/100)
	dim = (width, height)
	resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

	blur = cv2.blur(resized, (5,5))

	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

	# define range of blue color in HSV
	lower_green = np.array([20,40,40])
	upper_green = np.array([85,255,255])

	thresh = cv2.inRange(hsv, lower_green, upper_green)

	# morphologycal operation kernel definition
	kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
	sure_bg = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
	ret, sure_fg = cv2.threshold(dist_transform,0.05*dist_transform.max(),255,0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)

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

	labels_ori = cv2.watershed(resized,labels_ori)
	resized[labels_ori == -1] = [255,0,0]
	
	for i in range(len(greatest_areas)):
		obj = greatest_areas[i]
		#x = stats[obj,0]
		#y = stats[obj,1]
		j = stats[obj,2]
		k = stats[obj,3]
		fact = 0.2;

	# Draw centroids of selected objects
		resized = cv2.circle(resized, (int(centroids[obj,0]), int(centroids[obj,1])), 10, (255,0,0), 6)

	# show the output frame
	cv2.imshow("Frame", resized)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

