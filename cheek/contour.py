import numpy as np
import cv2
from matplotlib import pyplot as plt



def extract_cheeck_parts(img,points):
	mask = np.zeros(img.shape[0:2], dtype=np.uint8)

	points=np.array(points)
	points=points.reshape(1,len(points),2)
	    
	#method 1 smooth region
	cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

	#method 2 not so smooth region
	# cv2.fillPoly(mask, points, (255))
	 
	res = cv2.bitwise_and(img,img,mask = mask)
	rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
	cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
	 
	## crate the white background of the same size of original image
	wbg = np.ones_like(img, np.uint8)*255
	cv2.bitwise_not(wbg,wbg, mask=mask)
	# overlap the resulted cropped image on the white background
	dst = wbg+res
	 
	#plt.imshow(img,cmap='gray')
	#plt.show()
	#plt.imshow(mask,cmap='gray')
	#plt.show()
	#plt.imshow(cropped,cmap='gray')

	#plt.show()
	#plt.imshow(res,cmap='gray')
	#plt.show()
	plt.imshow(dst,cmap='gray')
	plt.show()
