
#import library
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

src_path = 'rahul.jpg'

img = cv2.imread(src_path, 0)   # 0 imports a grayscale

frangi_img=frangi(img, scale_range=(1, 10), scale_step=1, black_ridges=True)#(scale_range=(1, 10), scale_step=2, alpha=0.5, beta=0.5, frangi_c=500, black_vessels=True)

#plt.imshow(frangi_img,cmap='hot')
#plt.show()

sc=StandardScaler() #scalling

gray_image=sc.fit_transform(frangi_img)

gray_image=gray_image*255 #range 0-255
binary_image=np.where(gray_image < 50//2, 0, 1) #change this range 

per=(binary_image.sum()*100)/(binary_image.shape[0]*binary_image.shape[1])
print('percentage of without dilation={}%'.format(per))

#plt.imshow(binary_image,cmap='gray')
#plt.show()

kernel = np.ones((5,5),np.float32) #remove noise
opening = cv2.morphologyEx(np.float32(binary_image), cv2.MORPH_OPEN, kernel)
#plt.imshow(opening,cmap='gray')

per=(opening.sum()*100)/(opening.shape[0]*opening.shape[1])
print('percentage of without dilation={}%'.format(per))

