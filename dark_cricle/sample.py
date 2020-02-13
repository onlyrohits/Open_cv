import cv2 
from matplotlib import pyplot as plt
import numpy as np
import imutils

import utils

face = cv2.imread('face.jpg')
 
face=np.where(face<255,face,0)

B, G, R = cv2.split(face) 
# Corresponding channels are seperated 



eye = cv2.imread('eye.jpg')
 
eye=np.where(eye<255,eye,0)

B1, G, R = cv2.split(eye) 
# Corresponding channels are seperated 





print(np.average(B1)+np.average(B))
















