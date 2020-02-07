
# Python program to explain cv2.ellipse() method 
	
# importing cv2 
import imutils
import cv2 
import matplotlib.pyplot as plt
import numpy as np



def dark_circle(img,cordinate):

    image=img



    image_copy=np.copy(image)

    mask = np.zeros(image.shape[0:2], dtype=np.uint8)


    center_coordinates = cordinate

    axesLength = (60, 60) #(x,y)

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

    #plt.imshow(image,cmap='gray')
    #plt.show() 


    #plt.imshow(image_mask,cmap='gray')
    #plt.show() 


    image_copy[image_mask==0]=0


    #print("avg_pixels=",np.average(image_copy))

    cv2.imwrite('eye.jpg',image_copy)

    #avg_pixels(image_copy,image_mask)


    plt.imshow(image_copy,cmap='gray')
    plt.show() 

    return np.average(image_copy)


def avg_pixels(image_copy,image_mask):
	
	print(np.sum(image_copy))
	print(np.sum(image_mask))

	print(np.sum(image_copy)/np.sum(image_mask))