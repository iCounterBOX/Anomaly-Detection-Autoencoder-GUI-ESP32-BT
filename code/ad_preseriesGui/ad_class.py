# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 08:21:01 2024
Class to hold all kind of tools

@author: kristina
"""


import datetime
import cv2
import tensorflow as tf


class ad_class:        
    def __init__(self):
        self.maxScoreItem = []
        
        self.referenceList = []   # setzen wir mit dem ini file
        self.finalResultLst = []
      
      
    def dt(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - '

    def showInMovedWindow(self,  winname, img, x, y):
        cv2.namedWindow(winname)        # Create a named window
        cv2.moveWindow(winname, x, y)   # Move it to (x,y)...THis way the image ma appear on TOP of other screens!
        cv2.imshow(winname,img)   
    
      
    def msgWithTextOnImage(self,  winname, imgTxt, img, x, y, farbe):
        cv2.namedWindow(winname)        # Create a named window
        cv2.moveWindow(winname, x, y)   # Move it to (x,y)...THis way the image ma appear on TOP of other screens!
        
        # Write some Text

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,100)
        fontScale              = 2
        fontColor              = farbe  #(50,205,50)
        thickness              = 3
        lineType               = 2

        cv2.putText(img,imgTxt , 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        cv2.imshow(winname,img)
        
    # count how many layers my model has    
    #https://stackoverflow.com/questions/48265926/keras-find-out-the-number-of-layers    
    def count_layers(self, model):
       num_layers = len(model.layers)
       for layer in model.layers:
          if isinstance(layer, tf.keras.Model):
             num_layers += self.count_layers(layer)
       return num_layers
    
    