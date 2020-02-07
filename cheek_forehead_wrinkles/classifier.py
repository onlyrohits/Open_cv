
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
x = tf.placeholder("float", None)

#from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#load pickle file
import pickle

forehead_pkl=open('forehead_classification.pkl','rb')
model= pickle.load(forehead_pkl)





"""
testing=cv2.imread('forehead.jpg',cv2.IMREAD_GRAYSCALE)

#plt.imshow(testing)

testing=np.resize(testing,(28,28))
# Prepare the training images
testing = testing.reshape(1,28,28, 1)
testing = testing.astype('float32')
testing /= 255

#%%time
pred=model.predict(testing)

if pred.argmax(1)[0]==1:
  print("clean Face")
else:
  print("wrinkled face")
"""
