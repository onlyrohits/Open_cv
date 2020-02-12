from imutils import face_utils
import imutils
import numpy as np
import collections
import matplotlib.pyplot as plt
import dlib
import cv2

from crop import crop_forehead,extract_cheeck_parts




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
rects = detector(gray, 1)

# loop over the face detections

points=[]

left_points=[]
right_points=[]

total_points=[]


right_cheek=[0,1,2,3,4,28]
left_cheek=[12,13,14,15,16,28]


mark=[18,20,25,27] #eyes cordinates that we will pickup points

for (i, rect) in enumerate(rects):

	shape_ = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape_)

	landmark = np.empty([68, 2], dtype=int)
	for i in range(68):
		landmark[i][0] = shape_.part(i).x
		landmark[i][1] = shape_.part(i).y

		if i in mark:
			#print("(x,y)=",(landmark[i][0],landmark[i][1])) #get the cordinate of topmost eyes point
			#print(landmark[i][1])
			print(i)
			points.append((landmark[i][0],landmark[i][1]))

		elif i in right_cheek:
			right_points.append([landmark[i][0],landmark[i][1]])
		elif i in left_cheek:
			left_points.append([landmark[i][0],landmark[i][1]])



left_points.append(right_points[-1]) #appned last 28 landmark in left points
total_points.extend(right_points)
total_points.extend(left_points)
#print(total_points)

per1=crop_forehead(gray,points) #call function that return forehead cordinate


per2=extract_cheeck_parts(gray,total_points)

print("total wrinkles=",(per1+per2))


#print("forehead top cordinate=",crop_cordinate)






