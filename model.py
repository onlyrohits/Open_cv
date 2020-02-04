from imutils import face_utils
import imutils
import numpy as np
import collections
import matplotlib.pyplot as plt
import dlib
import cv2

from contour import dark_circle
#from crop import crop_forehead




"""
MAIN CODE STARTS HERE
"""
# load the input image, resize it, and convert it to grayscale

img_digit=input('Enter Image digit >> ')

img_name='img{}.jpg'.format(img_digit)

image = cv2.imread(img_name)

image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

out_face = np.zeros_like(image)

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# detect faces in the grayscale image
rects = detector(gray, 0)

# loop over the face detections



 #eyes cordinates that we will pickup points

for (i, rect) in enumerate(rects):

	shape_ = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape_)

	landmark = np.empty([68, 2], dtype=int)
	for i in range(68):
		landmark[i][0] = shape_.part(i).x
		landmark[i][1] = shape_.part(i).y

		if i==41 or i==46:
			print('i=',i,(landmark[i][0],landmark[i][1]))
			dark_circle(gray,(landmark[i][0],landmark[i][1]))



