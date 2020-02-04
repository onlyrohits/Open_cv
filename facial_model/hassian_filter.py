
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

#https://datacarpentry.org/image-processing/07-thresholding/
import numpy as np
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer

src_path = 'rahul.jpg'

def detect_ridges(gray, sigma=10.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges


img = cv2.imread(src_path, 0) # 0 imports a grayscale
if img is None:
    raise(ValueError(f"Image didn\'t load. Check that '{src_path}' exists."))

a, b = detect_ridges(img, sigma=10.0)




plt.imshow(a,cmap='gray')

plt.imshow(b,cmap='gray')

blur = skimage.color.rgb2gray(a)

# perform adaptive thresholding
t = skimage.filters.threshold_otsu(blur)
mask = blur > t

#convert a boolean array to an int array
binary_array=mask*1

print(binary_array)
#find percentage
binary_array=binary_array.astype(float)
per=(binary_array.sum()*100)/(binary_array.shape[0]*binary_array.shape[1])

print("percentage=",per)

plt.imshow(mask,cmap='gray')

