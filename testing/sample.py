import cv2 
from matplotlib import pyplot as plt
import numpy as np
import imutils

"""
x=np.random.randint(5, size=(2, 40))
print(x)

print(np.mean(x))
print(np.average(x))
"""





"""
Let's crop the first 1000 rows(The first 5 rows in your image are white
 so the average will only be 255).
"""


img=cv2.imread('img8.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert it to RGB channel
plt.imshow(img,cmap='gray')
plt.show()

print("total rgb avg=",np.average(img))

print("avg on pixels r,g,b",np.average(img, axis = (0,1)))



img=cv2.imread('face.jpg')

#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img=np.where(img<255,img,0) #remove whites pixels
 #convert it to RGB channel
plt.imshow(img,cmap='gray')
plt.show()

print("total rgb avg=",np.average(img))

print("avg on pixels r,g,b",np.average(img, axis = (0,1)))






eye=cv2.imread('eye.jpg')

eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY) #convert it to RGB channel
plt.imshow(eye)
plt.show()



print("total rgb avg=",np.average(eye))

print("avg on pixels r,g,b",np.average(eye, axis = (0,1)))




















