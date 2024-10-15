# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 08:21:01 2024
Class to hold all kind of tools

@author: kristina
"""
from PyQt5.QtWidgets import  QMessageBox
import logging
import datetime
import numpy as np
import os
import cv2
import traceback


class tools:        
    def __init__(self):
        #kind of global for draw_rectangle() / a mouse callBack where we cannot give parameters in Method
        self.drawing = False
        self.ix, self.iy = -1,-1
        self.ixx, self.iyy = -1,-1
        self.img2 = None
        self.capFrame = None
        print("CV2 in class tools loaded")   
        
        
    
    '''
    MsgBox von pyQt / https://doc.qt.io/qt-6/qmessagebox.html  / - OK  Cancel
    '''
    def msgBoxInfoOkCancel(self,txt,title):
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - '
        logging.info(dt + 'msgBoxInfoOkCance()')
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)           
        msgBox.setText(txt)
        msgBox.setWindowTitle(title)
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)  
        v = msgBox.exec()
        return v
    
    def msgBoxYesCancel(self,txt,title):
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - '
        logging.info(dt + 'msgBoxYes()')
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)           
        msgBox.setText(txt)
        msgBox.setWindowTitle(title)
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        return msgBox.exec()
    
    def msgBoxYesNoCancel(self,txt,title):
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - '
        logging.info(dt + 'msgBoxYesNoCancel()')
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)           
        msgBox.setText(txt)
        msgBox.setWindowTitle(title)
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        return msgBox.exec()
    
    def putMsgInCv2Image(self, imgTxt, img, farbe):       
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,100)
        fontScale              = 5
        fontColor              = farbe  #(50,205,50)
        thickness              = 4
        lineType               = 2

        cv2.putText(img,imgTxt , 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        # text is now IN the image
        return img
         
    
    
    '''
    ADD Text (in)to an image
    https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
FONT_HERSHEY_COMPLEX
FONT_HERSHEY_COMPLEX_SMALL
FONT_HERSHEY_DUPLEX
FONT_HERSHEY_PLAIN
FONT_HERSHEY_SCRIPT_COMPLEX
FONT_HERSHEY_SCRIPT_SIMPLEX
FONT_HERSHEY_SIMPLEX
FONT_HERSHEY_TRIPLEX
FONT_ITALIC
    '''
    def draw_text(self, img, text,
              font=cv2.FONT_HERSHEY_DUPLEX,
              pos=(0, 0),
              font_scale=6,
              font_thickness=4,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):

        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        #cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

        return  img

        
    def dt(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - '
    
    def testDevice(self, webcamNr):
       logging.info(self.dt() + 'testDevice() - we now check device: ' + str(webcamNr))
       cap = cv2.VideoCapture(webcamNr)         
       if cap is None or not cap.isOpened():
           print('Warning: unable to open video source: ' + str(webcamNr))
           logging.info(self.dt() + 'Warning: unable to open video source: ' + str(webcamNr))
           return False
       else:
           print('YEAA: this video source seem ok: ' + str(webcamNr))
           logging.info(self.dt() + 'YEAA: this video source seem ok: ' + str(webcamNr))
           return True 


    # delete file if exist
    def removeFile(self, filePath):
        # check whether the file exists
        try:
            if os.path.exists(filePath):
                # delete the file
                os.remove(filePath)
        except Exception as e:
            logging.error(traceback.format_exc())
            print(e)
        
        

    def showInMovedWindow(self,  winname, img, x, y):
        cv2.namedWindow(winname)        # Create a named window
        cv2.moveWindow(winname, x, y)   # Move it to (x,y)...THis way the image ma appear on TOP of other screens!
        cv2.imshow(winname,img)
 
    # FILTER   FILTER   FILTER   FILTER   FILTER   FILTER  FILTER   FILTER    
 
    
    

    def contourFilter1(self, imgOriPath, originalImg ):
        imgFiltered = cv2.imread(imgOriPath,0) # gray img             
       
        #canny
        sigma = 0.33
        median = np.median(cv2.imread(imgOriPath))
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        imgFiltered = cv2.Canny(originalImg , lower, upper)
        
        
        #THRESOULD
        #_, binary = cv2.threshold(imgFiltered, self.contour_theshValue, 255, cv2.THRESH_BINARY)
                  
        #imgFiltered = binary
        
        cnts,_ = cv2.findContours(imgFiltered , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print ( "nr contours: {}".format(len(cnts)) )
        oimg = cv2.imread(imgOriPath)
        
        # create a white image
       
        img = np.ones((640, 480, 3), dtype = np.uint8)
        imgW = 255* img
        
        #imgW = Image.new('RGB', (_oriImgWidth, _oriImgHeight), color = (255,255,255))
        for c in cnts:
            area = cv2.contourArea(c)
            print("area: " + str(area))
            if area > 130 :
                cv2.drawContours( imgW, [c], 0, (0,0,0), -1)
        #cv2.drawContours(oimg, cnts, -1, (0,255,0), 3)
        return imgW
        
    
        
        