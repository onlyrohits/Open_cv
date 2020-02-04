import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import json



for i in range(0,2):
    df=pd.read_csv('data.csv')
    df=df.iloc[i,:]  
    
    cordinate=json.loads(df[5])
    
    
    x=[[cordinate['all_points_x'][i],cordinate['all_points_y'][i]] for i in range(0,len(cordinate['all_points_x']))]
    
    x=np.array(x)
    
    print(x)
    x=x.reshape(1,len(x),2)
    
    
    
    img = cv2.imread('facial_landmarks.jpg')
    
    
    
    cv2.fillPoly( img,x, 255 )
    
    plt.imshow(img)
    plt.show()




