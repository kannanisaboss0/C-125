#------------------------------C-125----------------------------#
#-------------------------ImagePredictor.py----------------------------#

'''
Importing modules:
-numpy (np)
-pandas (pd)
-cv2 
-Logistic Regression (LogReg) :-sklearn.linear_model
-train_test_split (TTs) :-sklearn.model_selection
-accuracy_score (a_s) :-sklearn.metrics
-sys
-webcolors (wb)
-random (rd)
-time (tm)
-Image :-PIL
-PIL.ImageOps
'''

import numpy as np 
import pandas as pd 
import cv2
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import train_test_split as TTs
from sklearn.metrics import accuracy_score as a_s
import sys
import webcolors as wb
import random as rd
import time as tm
from PIL import Image
import PIL.ImageOps

#Sourcing the values
Y_val=pd.read_csv("data.csv")
X_val=np.load("image.npz")["arr_0"]

#Splitting the data into two sections for testing and training
tm.sleep(0.2)
X_train,X_test,Y_train,Y_test=TTs(X_val,Y_val,train_size=9999,test_size=1,random_state=9)

#Scaling the train and test values of the X variable
X_train=X_train/255
X_test=X_test/255

#Initiating the Logistic Regression classifier and modelling  the data
LR=LogReg(solver="saga",multi_class="multinomial")
LR.fit(X_train,Y_train)

#Predicting the values
Y_prediction=LR.predict(X_test)

#Yielding the accuracy score and printing it
accuracy_sc=a_s(Y_test,Y_prediction)
print("Overall accuracy is {}%".format(accuracy_sc*100))

#Definng a function to recieve an image as a parameter and resultantly prescribing a prediction --~> used in ToApI.py
def emulate_predictor(image_param):
    #Optimizing the image to increase prediciton accurrcay
    image=Image.open(image_param)
    image=image.convert('L')
    image=image.resize((22,30),Image.ANTIALIAS)
    px_flt=20
    min_px=np.percentile(image,px_flt)
    image=np.clip(image-min_px,0,255)
    max_px=np.max(image)
    image=np.asarray(image)/max_px

    #Reshaping the array
    value=np.array(image).reshape(1,660)

    #Prediciting the values
    prediction=LR.predict(value)

    #Returning the first element of the prediction
    return prediction[0]

#-------------------------ImagePredictor.py----------------------------#
#------------------------------C-125----------------------------#
