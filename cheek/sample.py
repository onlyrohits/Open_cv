import numpy as np
import cv2
import imutils
import cv2 
import matplotlib.pyplot as plt
import numpy as np

 
 
 
img = cv2.imread("img2.jpg")
img = imutils.resize(img, width=500)

img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = np.zeros(img.shape[0:2], dtype=np.uint8)
points = np.array([[[28, 121], [33, 185], [40, 246], [52, 306], [74, 365], [252, 163], [446, 370], [467, 312], [478, 252], [486, 193], [491, 131], [252, 163]]])
 
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
