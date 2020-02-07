import cv2 
import imutils
import cv2
import matplotlib.pyplot as plt
from skimage.filters import frangi, hessian
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import numpy as np
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer
from sklearn.preprocessing import  MinMaxScaler,StandardScaler
from skimage.color import rgb2gray

from frangi import frangi_filter



def crop_forehead(img_array,cordinates):
    
	img=img_array

	points=cordinates

	x1=points[0][0]
	y1=points[1][1]    
	x2=points[3][0]
	y2=points[2][1]

	dist=np.sqrt(np.square(x1-x2)+np.square(y1-y2))

	#creating topmost corner

	x1_new=x1

	y1_new=int(y1-dist)

	width=int(x1_new+dist)

	height=int(y1_new+dist)

	forehead = img[y1_new+20:height-20,x1_new:width+50] #crop images

	#plt.imshow(forehead)
	#cv2.imwrite('forehead.jpg',forehead)

	per=frangi_filter(forehead) #pass image to frangi filter and get percentage
	#return [(x1_new,y1_new),(width,height)],per

	print('percentage of Wrinkles on forehead={}%'.format(per))


	return per


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

	per=frangi_filter(dst)
	#cv2.imwrite('cheek.jpg',dst)

	#plt.imshow(dst,cmap='gray')
	#plt.show()
	#print(per)
	print('percentage of Wrinkles on cheek={}%'.format(per))

	return per



