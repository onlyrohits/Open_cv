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
	return [(x1_new,y1_new),(width,height)],per


def frangi_filter(forehead):


	img=forehead
	img = imutils.resize(img, width=800)
	#img = rgb2gray(img)

	cv2.imwrite('forehead.jpg',img)

	frangi_img=frangi(img, sigmas=(1, 10), scale_step=1, black_ridges=True)#(scale_range=(1, 10), scale_step=2, alpha=0.5, beta=0.5, frangi_c=500, black_vessels=True)

	#plt.imshow(img,cmap='hot')
	#plt.show()
	cv2.imwrite('frangi.jpg',frangi_img)

	sc=StandardScaler() #scalling

	gray_image=sc.fit_transform(frangi_img)

	gray_image=gray_image*255 #range 0-255

	binary_=np.where(gray_image < 50//2, 0, 255)


	binary_image=np.where(gray_image < 50//2, 0, 1) #change this range 

	#cv2.imwrite('binary.jpg',binary_)

	per=(binary_image.sum()*100)/(binary_image.shape[0]*binary_image.shape[1])
	print('percentage of without dilation={}%'.format(per))

	#plt.imshow(binary_image,cmap='gray')
	#plt.show()
	#cv2.imwrite('binary_image.jpg',binary_image)

	kernel = np.ones((5,5),np.float32) #remove noise
	opening = cv2.morphologyEx(np.float32(binary_image), cv2.MORPH_OPEN, kernel)

	#cv2.imwrite('opening.jpg',opening)

   # plt.imshow(opening,cmap='gray')

	per=(opening.sum()*100)/(opening.shape[0]*opening.shape[1])
	print('percentage of without dilation={}%'.format(per))

	return per