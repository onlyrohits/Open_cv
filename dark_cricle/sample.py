# Python program to explain cv2.ellipse() method 
	
# importing cv2 
import imutils
import cv2 
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread('img7.jpg')

image = imutils.resize(image, width=500)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




image_copy=np.copy(image)

mask = np.zeros(image.shape[0:2], dtype=np.uint8)


center_coordinates = (173, 415)


axesLength = (60, 40) #(x,y)

angle = 0

startAngle = 10

endAngle = 170

# Red color in BGR 
color = (255, 255, 255) 

# Line thickness of 5 px 
thickness = -1

# Using cv2.ellipse() method 
# Draw a ellipse with red line borders of thickness of 5 px 
image_mask = cv2.ellipse(mask, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness) 

# Displaying the image 
plt.imshow(image_mask,cmap='gray')
plt.show() 


image_copy[image_mask==0]=0

plt.imshow(image,cmap='gray')
plt.show() 




