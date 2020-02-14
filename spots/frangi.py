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
	per=frangi_filter(dst)
	cv2.imwrite('cheek.jpg',dst)

	#plt.imshow(dst,cmap='gray')
	#plt.show()
	print(per)


def frangi_filter(forehead):


	img=forehead
	#img = rgb2gray(img)

	#cv2.imwrite('forehead.jpg',img)

	frangi_img=frangi(img, sigmas=(1, 10), scale_step=1, black_ridges=True)#(scale_range=(1, 10), scale_step=2, alpha=0.5, beta=0.5, frangi_c=500, black_vessels=True)

	#plt.imshow(img,cmap='hot')
	#plt.show()
	#cv2.imwrite('frangi.jpg',frangi_img)

	sc=StandardScaler() #scalling

	gray_image=sc.fit_transform(frangi_img)

	gray_image=gray_image*255 #range 0-255

	binary_=np.where(gray_image < 50//2, 0, 255)


	binary_image=np.where(gray_image < 50//2, 0, 1) #change this range 

	#cv2.imwrite('binary.jpg',binary_)

	per=(binary_image.sum()*100)/(binary_image.shape[0]*binary_image.shape[1])
	#print('percentage of without dilation={}%'.format(per))

	#plt.imshow(binary_image,cmap='gray')
	#plt.show()
	#cv2.imwrite('binary_image.jpg',binary_image)

	kernel = np.ones((5,5),np.float32) #remove noise
	opening = cv2.morphologyEx(np.float32(binary_image), cv2.MORPH_OPEN, kernel)

	#cv2.imwrite('opening.jpg',opening)

   # plt.imshow(opening,cmap='gray')

	per=(opening.sum()*100)/(opening.shape[0]*opening.shape[1])
	#print('percentage of Wrinkles={}%'.format(per))

	return per