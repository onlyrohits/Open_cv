#https://answers.opencv.org/question/121657/get-roi-from-face-landmark-points-cv2-dlib/
from collections import OrderedDict
import numpy as np
import cv2
import dlib
import imutils
from matplotlib import pyplot as plt


CHEEK_IDXS = OrderedDict([("right_cheek", (0,1,2,3,4,28)),
                        ("left_cheek", (12,13,14,15,16,28))
                         ])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img = cv2.imread('img2.jpg')
img = imutils.resize(img, width=500)

overlay = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

points=[]

detections = detector(gray, 0)
for k,d in enumerate(detections):
    shape = predictor(gray, d)
    for (_, name) in enumerate(CHEEK_IDXS.keys()):
        pts = np.zeros((len(CHEEK_IDXS[name]), 2), np.int32) 
        for i,j in enumerate(CHEEK_IDXS[name]): 
            pts[i] = [shape.part(j).x, shape.part(j).y]
            points.append( [shape.part(j).x, shape.part(j).y])
        pts = pts.reshape((-1,1,2))
        cv2.polylines(overlay,[pts],True,(0,255,0),thickness =2)


    plt.imshow(overlay)
    plt.show()
    
print(points)    