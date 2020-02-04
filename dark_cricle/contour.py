
# Python program to explain cv2.ellipse() method 
	
# importing cv2 
import imutils
import cv2 
import matplotlib.pyplot as plt
	



def dark_circle(img,cordinate):

    image=img
    center_coordinates = cordinate
    
    axesLength = (60, 40) #(x,y)
    
    angle = 0
    
    startAngle = 10
    
    endAngle = 170
    
    # Red color in BGR 
    color = (0, 0, 255) 
    
    # Line thickness of 5 px 
    thickness = -1
    
    # Using cv2.ellipse() method 
    # Draw a ellipse with red line borders of thickness of 5 px 
    image = cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness) 
    
    # Displaying the image 
    plt.imshow(image)
    plt.show() 
