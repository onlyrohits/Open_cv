# Python program to explain cv2.ellipse() method 
	
# importing cv2 
import imutils
import cv2 
import matplotlib.pyplot as plt
	
# path 
path = 'img2.jpg'

	
# Reading an image in default mode 
image = cv2.imread(path) 
image = imutils.resize(image, width=500)
	
# Window name in which image is displayed 
window_name = 'Image'

center_coordinates = (318, 132) 

axesLength = (100, 50) 

angle = 0

startAngle = 0

endAngle = 180

# Red color in BGR 
color = (0, 0, 255) 

# Line thickness of 5 px 
thickness = 5

# Using cv2.ellipse() method 
# Draw a ellipse with red line borders of thickness of 5 px 
image = cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness) 

# Displaying the image 
plt.imshow(image)
plt.show() 
