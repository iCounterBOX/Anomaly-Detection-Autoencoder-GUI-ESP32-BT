# -*- coding: utf-8 -*-
"""
Created 08.07.24

@author: kristina

movie2cartoon  - PY38 / CONDA / anaconda / Spyder

    app öffnet die USB-Cam
    recorded die cam und wirft filter drauf
    output ist dann ein movie im cartoon-style
    
    hier kann zw. mp4 video oder cam umgeschaltet werden:
        # create video capture - webCam Higher resolution
        '''
        self.cap = cv2.VideoCapture(int(self.lineEdit_camNr.text())  ,cv2.CAP_DSHOW )
        self.cap.set(3,1280)
        self.cap.set(4,720)
        '''
        self.cap = cv2.VideoCapture(myVideo)  

23.07.24:
    movie2cartoon is going to github
    requirements.txt : ... is generating file in the working directory from movie2Cartoon
        pipreqs --encoding utf-8 "./"
    ..incase of cloning the repo..late again in the working folder:
        pip install -r requirements.txt     
    
"""


from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import (
    QMessageBox, 
    QListWidget, 
    QPushButton, 
    QComboBox, 
    QCheckBox, 
    QLabel,
    QLineEdit,
    )
from PyQt5.QtGui import QImage
'''
ONLY import the UI resource file here - https://stackoverflow.com/questions/50627220/loadui-loads-everything-but-no-images-in-pyqt
pyrcc5 -o image_rc.py image.qrc
Issue: without..No Images will be displayed!!
'''
import image_rc  

import sys
import time
import os
import cv2
import traceback
import logging
import pandas as pd

#import MY own classes
from class_tools import tools
toolz = tools()  

'''
https://stackoverflow.com/questions/7484454/removing-handlers-from-pythons-logging-loggers
Zäh! einmal angelegt wird der name dem Handler übergeben..das auch NUR in einem unserer Module!
Im notfall wenn sich mal der namen des log ändern sollte, dann den Handler rücksetzen:
        
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])
    
    logFileName = os.getcwd() + "\\" +"companyTasks.log"
    logger = logging.getLogger(logFileName)
    logging.basicConfig(filename='companyTasks.log', encoding='utf-8', level=logging.INFO)
    logger.debug('This message should go to the log file')
    logger.info('So should this')
    logger.warning('And this, too')
    logger.error('And non-ASCII stuff, too, like Øresund and Malmö')
'''

logFileName = os.getcwd() + "\\" +"filterDash.log"
logger = logging.getLogger(logFileName)
#logging.basicConfig(filename='companyTasks.log', encoding='utf-8', level=logging.INFO)
logging.basicConfig(filename = logFileName,  level=logging.INFO)

logging.info(toolz.dt() + '******** This is Module einrichtungMain.py  GO ***************************')

print(cv2.__file__) 
print (cv2. __version__ )
logging.info(toolz.dt() + cv2. __version__ )

from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
print("Qt: v", QT_VERSION_STR, "\tPyQt: v", PYQT_VERSION_STR)



_UI_FILE = os.path.join(os.getcwd(),"filterUI.ui" )
logging.info(toolz.dt() + 'UI file: ' + _UI_FILE)
 
#myVideo = os.getcwd() + "\\" +"video/aliAkasoSmall.mp4"
 


class window(QtWidgets.QMainWindow):
    def __init__(self):
        #super(MainWindow_ocrTemplate,self).__init__()
        super().__init__()

        logging.info(toolz.dt() + 'super(window...init..done')
        # load ui file
        try:  
            uic.loadUi( _UI_FILE, self)
        except Exception as e:
            print(e)
            logging.error(traceback.format_exc())      
        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)   # added newly
       
        
        #bgRemove Filter
        self.horizontalSlider_thresh1.valueChanged['int'].connect(self.bgThresh_value) # type: ignore
        self.horizontalSlider_thresh1.setMaximum(255)
        
        self.horizontalSlider_blur.valueChanged['int'].connect(self.blur_valueFkt) # type: ignore
        self.horizontalSlider_blur.setMaximum(99)
        
        self.horizontalSlider_adaptivThresh1.valueChanged['int'].connect(self.fkt_adapThresh1) # type: ignore
        self.horizontalSlider_adaptivThresh1.setMaximum(255)
        self.horizontalSlider_adaptivThresh2.valueChanged['int'].connect(self.fkt_adapThresh2) # type: ignore
        self.horizontalSlider_adaptivThresh2.setMaximum(100)
        
        self.horizontalSlider_valueScale.valueChanged['int'].connect(self.valueScale_sliderValue) # type: ignore
        self.horizontalSlider_valueScale.setMaximum(100)
        
        #Buttons
        self.pushButton_deleteTrainImages.clicked.connect(self.deleteTrainImages)
        
        #kind of global
        self.bgRemove_thresh_value = 1   #0.5
        self.blur_value = 5
        self.adaptiveThreshValue1 = 21
        self.adaptiveThreshValue2 = 0.5
        self.value_scaleX = 1
        self._capFrameImage = cv2
        
        self._tic = time.perf_counter()
        self._toc = time.perf_counter()
        
        #nummerierung für den train material image filename
        self.nx = 0  
        self.trainImage_path = 'images4training'
        
        # create a timer  * TIMER 
        self.timer = QtCore.QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.pushButton_StartStopCam.clicked.connect(self.controlTimer)
        
        logging.info(toolz.dt() + 'show original image ..')
        
        self._objImgWindowWidth = 900
        self._objImgWindowHeight = 900
        
      
    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture - webCam Higher resolution
            
            self.cap = cv2.VideoCapture(int(self.lineEdit_camNr.text())  ,cv2.CAP_DSHOW )
            self.cap.set(3,1920) # MUSS rein sonst fällt er auf 640x480 zurück / 1280 x 720 / 1920 x 1080
            self.cap.set(4,1080)
            self.cap.set(5,30)
            
            #self.cap = cv2.VideoCapture(myVideo)  
            
            # We need to set resolutions.   https://www.geeksforgeeks.org/saving-a-video-using-opencv/
            # so, convert them from float to integer. 
            frame_width = int(self.cap.get(3)) 
            frame_height = int(self.cap.get(4))    
            size = (frame_width, frame_height) 
            self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.out = cv2.VideoWriter('output.mp4', self.fourcc, 10.0, size)# e.g. 30  - ACHTUNG mit Frametime
            
            # start timer  - FPS  !!! ACHTUNG mit FRame time from witer!! ...die 1000 ( 1 sec ) ist jetzt NUR wegen making von AI Training material making!!!
            self.timer.start(500)    #e.g. 10  or 30
            # update control_bt text
            self.pushButton_StartStopCam.setText("Stop Cam")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            self.out.release()
            # update control_bt text
            self.pushButton_StartStopCam.setText("Start Cam" )
            #self.ui.image_label.setText("Camera")

      
    
    def deleteTrainImages(self):
        dx = toolz.msgBoxYesNoCancel("Train Material Folder", "Delete all elements in Training-Folder?")
        if dx == QMessageBox.Yes:
            self.nx = 0 
            self.lcdNumber_lcd.display(self.nx)
            for f in os.listdir(self.trainImage_path):     
                os.remove(os.path.join(self.trainImage_path, f))
            
    
    # handle the red cross event
    def closeEvent(self, event): 
        try:
            self.started = False            
            logging.shutdown()
            self.cap.release()
            self.out.release()
            cv2.destroyAllWindows() # Closes all the frames 
            self.close()            
            #QtWidgets.QApplication.quit()
            #QtWidgets.QCoreApplication.instance().quit()            
            event.accept()
            print('Window closed')
            
            #sys.exit() # die beiden KILL python ALL - restart Kernel :-)
            
        except Exception as e:
            print(e)
            logging.error(traceback.format_exc())          
   
    
    # -----------------------  OpenCV-Filter  ----------------------------------------
    
    def bgThresh_value(self,value):
        #This function will take value from the slider  for the threshould from 0 to 99        
        self.bgRemove_thresh_value = value
        print('bgRemove_thresh_value: ',self.bgRemove_thresh_value)
        self.lineEdit_thresh1.setText(str(self.bgRemove_thresh_value))    
    
        #---- ADAPTIVE Thresh - start
    def blur_valueFkt(self,value):
        if value % 2 == 0:
            pass # Even 
        else:
           self.blur_value = value  # e.g 7      
        print('blur_valueFkt: ',str(self.blur_value))
        self.lineEdit_blur.setText(str(self.blur_value))     
     
    def fkt_adapThresh1(self,value):  
        #first value is only allowed odd
        if value % 2 == 0:
            pass # Even 
        else:
            self.adaptiveThreshValue1 = value # odd is ok..e.g.21      
        print('fkt_adapThresh1: ',str(value))
        self.lineEdit_value_adapThresh1.setText(str(value)) 
        
    def fkt_adapThresh2(self,value):         
        self.adaptiveThreshValue2 = value
        print('fkt_adapThresh2: ', str(value))
        self.lineEdit_value_adapThresh2.setText(str(value))  
        
        #---- ADAPTIVE Thresh - start
        
        
    def valueScale_sliderValue(self, value):
        self.value_scaleX = value/100
        print('scale factor : ',self.value_scaleX )
        self.lineEdit_valueScale.setText(str(self.value_scaleX ))  
     
    
    #------------------------------------------------------------------------------------
          
    
    # view camera
    def viewCam(self):
        # read image in BGR format
        ret, frame = self.cap.read()
        
        """measure time between frames"""
        self._tic = time.perf_counter()   # current time
        secs = self._tic - self._toc      # difference to last time
        print("time since last frame (seconds): ", str( round(secs, 2)))
        self._toc = self._tic

     
        if ret == True:       
            self.update (frame) # zum filter window
        
        
        framex = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(framex, framex.shape[1],framex.shape[0],framex.strides[0],QImage.Format_RGB888) 
        pixmap = QtGui.QPixmap.fromImage(img)
        pixmap = pixmap.scaled(self._objImgWindowWidth, self._objImgWindowHeight, QtCore.Qt.KeepAspectRatio)
        self.lbl_imgOriginal.setPixmap(pixmap)  
        
        
      
           
    def update(self, capFrame):
        """ This function will update the photo according to the 
            current values of blur and brightness and set it to photo label.
        """     
        # frame der cam  / zwischenspeichern und dem Filter übergeben     
        
        
        imgFiltered = capFrame.copy() 
        
        # LOOOP start     
        
        try:
            if self.checkBox_treshouldFilter.isChecked() == True:            
                # machen hier weg weg wegen cvzone etc issues
                pass
                
               
            if self.checkBox_adaptThresh.isChecked() == True:               
               # imgFiltered = toolz.contourFilter1(self.imgOriPath, self.originalImg )
               
               gray = cv2.cvtColor(imgFiltered, cv2.COLOR_BGR2GRAY)
               kernel_size = (self.blur_value+1, self.blur_value+1) # +1 is to avoid 0
               #blur = cv2.blur(gray, kernel_size)  
               
               kernel_size = (self.blur_value, self.blur_value) # odd?                    
               blur = cv2.GaussianBlur(gray, kernel_size, 0)
               
               
               #edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.adaptiveThreshValue1 ,self.adaptiveThreshValue2)
               
               edges = cv2.adaptiveThreshold(blur, 255,	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, self.adaptiveThreshValue1 ,self.adaptiveThreshValue2)
               # LESEN:   https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
               color = cv2.bilateralFilter(imgFiltered, 9, 250, 250)   # parametrisieren??
               imgFiltered = cv2.bitwise_and(color, color, mask=edges)
               
            
            #https://learnopencv.com/image-resizing-with-opencv/#resize-by-wdith-height
            if self.checkBox_valueScale.isChecked() == True:  
               if  self.value_scaleX == 0: self.value_scaleX = 0.1
               imgFiltered =  cv2.resize(imgFiltered, (0,0), fx=self.value_scaleX, fy=self.value_scaleX) # scaling OK 0..1
               #imgFiltered =  cv2.resize(originalImg, (self.value_scaleX, self.value_scaleX), interpolation = cv2.INTER_LINEAR) # selbe size wie original!
              
            self.out.write(imgFiltered)    # write as MP4 MOVIE!!!
            
            # write single images in a folder
            if self.checkBox_enableImageStoring.isChecked() == True:  
                base_path = os.path.join(self.trainImage_path, "trainImg")
                cv2.imwrite('{}_{}.{}'.format(base_path, self.nx, 'jpg'), imgFiltered)
                self.nx += 1
                self.lcdNumber_lcd.display(self.nx)
                self.lcdNumber_lcd.repaint()

            #show final filtered image   
            img = cv2.cvtColor(imgFiltered, cv2.COLOR_BGR2RGB)
            img = QImage(img, img.shape[1], img.shape[0], img.strides[0],QImage.Format_RGB888)        
            pixmap = QtGui.QPixmap.fromImage(img)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(self._objImgWindowWidth, self._objImgWindowHeight, QtCore.Qt.KeepAspectRatio)
                self.lbl_filteredImg.setPixmap(pixmap) 
        
        except Exception as e:
            print(e)
            
        
        
       # LOOP END 
           
       
    

#THIS sector is needed for stand alone mode 
        
def app():
    app = QtWidgets.QApplication(sys.argv)      
    win = window()
    win.show()    
    sys.exit(app.exec_())

app()   

