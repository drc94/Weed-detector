# CENTER OF SHAPE APPLIET TO PLANT CENTER DETECTION
# import the necessary packages
import argparse
import imutils
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
# load the image, convert it to grayscale, blur it slightly, and threshold it
image = [cv2.imread(file) for file in glob.glob("Images/*.jpeg")]

resized_images = plt.figure()
resized_list = []

contour_images = plt.figure()
  
columns = 5
rows = 3

for i in range(len(image)):
	scale_percent = 40
	width = int(image[i].shape[1]*scale_percent/100)
	height = int(image[i].shape[0]*scale_percent/100)
	dim = (width, height)
	resized_list.append(cv2.resize(image[i], dim, interpolation = cv2.INTER_AREA))
	#resized_images.add_subplot(rows,columns,i+1)
	#plt.imshow(resized_list[i])

	hsv = cv2.cvtColor(resized_list[i], cv2.COLOR_BGR2HSV)

	# define range of blue color in HSV
	lower_green = np.array([30,40,40])
	upper_green = np.array([75,255,255])

	thresh = cv2.inRange(hsv, lower_green, upper_green)
	"""cv2.imshow('Binary image', thresh)
	cv2.waitKey(0)"""

	# morphologycal operation kernel definition
	kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
	kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
	"""opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	cv2.imshow('Opened image', opening)
	cv2.waitKey(0)

	closing = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	cv2.imshow('Closed image', closing)
	cv2.waitKey(0)"""

	erode = cv2.erode(thresh, kernel_erode, iterations = 1)
	"""cv2.imshow('Eroded', erode) 
	cv2.waitKey(0)"""

	dilate = cv2.dilate(erode, kernel_dilate, iterations = 1)
	"""cv2.imshow('Dilated', dilate) 
	cv2.waitKey(0)""" 

	contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# Draw all contours 
	# -1 signifies drawing all contours 
	cv2.drawContours(resized_list[i], contours, -1, (0, 255, 0), 3) 
	contour_images.add_subplot(rows,columns,i+1)
	plt.imshow(resized_list[i])

	"""cv2.imshow('Contours', resized) 
	cv2.waitKey(0) """

	"""res = cv2.bitwise_and(resized, resized, mask= img_dilation)

	cv2.imshow('Res image', res)
	cv2.waitKey(0)"""

plt.show()

