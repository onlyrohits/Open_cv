import numpy as np
import cv2
from matplotlib import pyplot as plt



def extract_cheeck_parts(img,points):
    
    points=np.array(points)
    
    points=points.reshape(1,len(points),2)
        
    cv2.fillPoly(img,points, 255)
    
    plt.imshow(img)
    plt.show()




