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
        
08.08.24: THIS is a clone from _cam2trainImage.py - we now use this gui for the Anamolay detection.
          We will use the Sliders for the threshold-values of the Autoencoder
          
02.09.24: BlueTooth - ESP32 - Poti_Controler
        BT Read the docs - D:\ALL_DEVEL_VIP\DOC\Anaconda_PY_ObjectDetect_Tensor_ORANGE\How2_AnomalyDetection_EncoderDecoder_1.docx
    
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



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KernelDensity

import cv2
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import random

#import MY own classes
from class_tools import tools
toolz = tools()  

from configparser import ConfigParser
config = ConfigParser()


'''
Bluetooth  BT -  connect esp32 via BT to this anomaly detector
Mit win10 "Change Bluetooth settings" bekommen wir den BT comport serial
e.g. Com ports /  COM12 Outgoing ESP32Bluetooth 'ESP32SPP'
bzw. prüfem Handy: mit dem Serial-Bluetooth Terminal
'''
import serial
esp32 = "ESP32Bluetooth"
mac_address = ""



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


_UI_FILE = os.path.join(os.getcwd(),"adUI.ui" )
logging.info(toolz.dt() + 'UI file: ' + _UI_FILE)

_myIniFile = os.getcwd() + "\config.ini"
logging.info(toolz.dt() + 'inifile : ' + _myIniFile)

 


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
       
        
        # INI File    
        self._statusBarTxt = "Anomaly Detection Statusbar"
        
        self._densityBT = 0        #ESP32  - BT serial Poti values
        self._reconstructErrBT = 0
        
        '''
        Config file:
        https://stackoverflow.com/questions/19078170/python-how-would-you-save-a-simple-settings-config-file
        standard is webcam 0.. on NB the inbuild cam..1 is normally the external USB cam
        
        '''
        self._camNr = 1
        #AUTOENCODER THRESHOLD:
        self.threshDensity_value = 0  #  -------------------------------->  DENSITY     
        self.threshReconstrErr_value = 0.000   # typical 0.002 ... 0.005 e.g. > RECONSTRUCTION Error ( acurancy )
         
        try:    
            if not config.read(_myIniFile):
                with open(_myIniFile,'a'): pass  # if not exist..create
                config.read(_myIniFile)
                config.add_section('main')
                config.set('main', 'webCamNr', '1')
                config.set('main', 'densityThresh', '2300')
                config.set('main', 'reconstructionErr', '0.004')                
                with open('config.ini', 'w') as f:
                    config.write(f)   
            if config.read(_myIniFile):
                self._camNr = config.getint('main', 'webCamNr')
                self.threshDensity_value = config.getint('main', 'densityThresh')
                self.threshReconstrErr_value = config.getfloat ('main', 'reconstructionErr')
                
                #pre initialize camNr & density & recon err
                self.lineEdit_camNr.setText(str(self._camNr))
                self.lineEdit_threshDensity.setText(str(self.threshDensity_value)) 
                self.lineEdit_threshReconstrErr.setText(str(self.threshReconstrErr_value))   
                
            print("webCamNr : " + str(self._camNr)) 
            print("density : " + str(self.threshDensity_value )) 
            print("reconstr err : " + str(self.threshReconstrErr_value ))  
        except Exception as e:
             print(e)   
             logging.error(traceback.format_exc())     
             pass
        
        
        #bgRemove Filter
        #DENSITY & Reconstruntion threshold sliders
        self.doubleSpinBox_densityThreshold.valueChanged.connect(self.threshDensity_fkt) # 
        self.doubleSpinBox_densityThreshold.setDecimals(0)
        self.doubleSpinBox_densityThreshold.setValue(self.threshDensity_value) 
        self.doubleSpinBox_densityThreshold.setSingleStep(1) 
               
        
        self.doubleSpinBox_reconstructionErrThreshold.valueChanged.connect(self.reconstructionErr_fkt) # 
        self.doubleSpinBox_reconstructionErrThreshold.setDecimals(4)
        self.doubleSpinBox_reconstructionErrThreshold.setValue(self.threshReconstrErr_value) 
        self.doubleSpinBox_reconstructionErrThreshold.setSingleStep(0.0001) 
        
        
        
        #ESP32
        
        self.lineEdit_threshDensity_ESP32.setStyleSheet("""QLineEdit { background-color: blue; color: white }""")
        self.lineEdit_threshReconstrErr_ESP32.setStyleSheet("""QLineEdit { background-color: blue; color: white }""")
        
        
        #Buttons  BUTTONS  BUTTONS
       
        # webcam pause
        self.pushButton_pause.clicked.connect(self.togglePause)
        
        # save presets for density & recon Err
        self.pushButton_saveDensityThreshold.clicked.connect(self.saveDensityThresh2IniFile)
        self.pushButton_saveReconstructThresh.clicked.connect(self.saveReconstructThresh2IniFile)
        
        # Take current value from the POTI´s
        self.pushButton_takeDensityThreshold_ESP32.clicked.connect(self.takePotiDensityAsNewThresh)
        self.pushButton_takeReconstructThresh_ESP32.clicked.connect(self.takePotiReconErrAsNewThresh)        
        
        
        #set status TIP
        self.pushButton_saveDensityThreshold.setStatusTip("save DensityThreshold as new default" )
        self.pushButton_saveReconstructThresh.setStatusTip("save save Reconstruct Thresh as new default" )
        
        #kind of global   -   initial values
        
        
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
        
        # CREATE BT Timer - Bluetooth
        self.timerBT = QtCore.QTimer()
        # set timer timeout callback function
        self.timerBT.timeout.connect(self.getESP32_BluetoothPotiValues)
        # start the BT timer
        self.timerBT.start(20)    #e.g. 10  or 30
        
        
        # files / folders
        # define the output file name and format
        self._output_file = 'capFrame/myFrame.jpg'    
        
        logging.info(toolz.dt() + 'show original image ..')
        
                
        # WE LOAD THE SAVED and TRAINED ANOMALY - MODEL - gebaut mit: 1_anomaly_make_h5_Model.py

        self.encodeDecode_model = load_model('model_encodeDecode.h5')
        self.anomaly_model = load_model('model_anomalyFinal.h5')
        
        self.isPause = False
                
        # THIS is for the autoencoder
        #Size of our input images
        self.SIZE =  128
        self.batch_size = 64
        self.datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = self.datagen.flow_from_directory(
            'trainImages/errorFree_train/',
            target_size=(self.SIZE, self.SIZE),
            batch_size=self.batch_size,
            class_mode='input'
            )


        
        #Get encoded output of input images = Latent space / Model.predict_generator` is deprecated and will be removed in a future version. Please use `Model.predict`,
        self.encoded_images = self.anomaly_model.predict(self.train_generator)

        # Flatten the encoder output because KDE from sklearn takes 1D vectors as input
        self.encoder_output_shape = self.anomaly_model.output_shape #Here, we have 16x16x16
        self.out_vector_shape = self.encoder_output_shape[1]*self.encoder_output_shape[2]*self.encoder_output_shape[3]

        self.encoded_images_vector = [np.reshape(img, (self.out_vector_shape)) for img in self.encoded_images]

        #Fit KDE to the image latent data -  D A U E R T   -   T I M E   c o n s u m i n g 
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(self.encoded_images_vector)  # <<<<<<< KDE
        
        
        #BT BLUETOOTH   BT BLUETOOTH   BT BLUETOOTH
        try:
            self._SerialPort = serial.Serial(
                port="COM12", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
            )
            sys.stdout.write("Python terminal emulator \n")
            print("SERIAL BT in ON !!!!!!!!!!")
        except Exception as e:
             print(e) 
             #toolz.msgBoxInfoOkCancel(str(e), "Bluetooth Issue")      
             pass

        self.statusbar.showMessage ("Ready to show some Detail-Data in THIS statusBar :-) ")
        
      
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
            self._SerialPort.close()   # CLOSE BT serial com-port
            self.timerBT.stop()  
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
   
    
   
    def togglePause(self):
        self.isPause = not self.isPause
        if self.isPause:            
            self.pushButton_pause.setText("Pause! -  click to continue!")
        if not self.isPause:            
            self.pushButton_pause.setText("Running -  click for Pause!")
            
        
    def saveDensityThresh2IniFile(self):
        config.set('main', 'densityThresh', self.lineEdit_threshDensity.text())
        with open('config.ini', 'w') as f:
            config.write(f)   
    
    def saveReconstructThresh2IniFile(self):
        config.set('main', 'reconstructionErr', self.lineEdit_threshReconstrErr.text())
        with open('config.ini', 'w') as f:
            config.write(f)  
    
    
    # -----------------------  THRESHOLD - SCHWELLWERT  ----------------------------------------
    
    def threshDensity_fkt(self,value):             
        self.threshDensity_value =  int(value)
        print('density_threshold value: ',str(self.threshDensity_value))
        self.lineEdit_threshDensity.setText(str(self.threshDensity_value))  
    
    def reconstructionErr_fkt(self,value):
        value = round(value,4)  
        self.threshReconstrErr_value = float(value)
        print('Reconstruction_Err threshold value: ',str(self.threshReconstrErr_value))
        self.lineEdit_threshReconstrErr.setText(str(self.threshReconstrErr_value))  
      
    def takePotiDensityAsNewThresh(self):
        self.doubleSpinBox_densityThreshold.setValue(int(self._densityBT)) 
        self.threshDensity_fkt(self._densityBT)
        
    
    def takePotiReconErrAsNewThresh(self):
        self.doubleSpinBox_reconstructionErrThreshold.setValue(float(self._reconstructErrBT )) 
        self.reconstructionErr_fkt(float(self._reconstructErrBT ))
    
    
    
    #------------------------------------------------------------------------------------
          
    
    # view camera
    def viewCam(self):
        # read image in BGR format
        ret, frame = self.cap.read()
        
        """measure time between frames"""
        self._tic = time.perf_counter()   # current time
        secs = self._tic - self._toc      # difference to last time
        #print("time since last frame (seconds): ", str( round(secs, 2)))
        self._toc = self._tic

     
        if ret == True:    
            # save the frame as an image file
            is_ok = cv2.imwrite(self._output_file, frame)
            if is_ok and not self.isPause:
                self.check_anomaly()   # <-------------  Check ANOMALY
                
        
        framex = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(framex, framex.shape[1],framex.shape[0],framex.strides[0],QImage.Format_RGB888) 
        pixmap = QtGui.QPixmap.fromImage(img)
        pixmap = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        self.lbl_liveWebCam.setPixmap(pixmap)   
       
    
        
    
    #Now, input unknown images and sort as Good or Anomaly
    def check_anomaly(self):
        density_threshold = int(self.lineEdit_threshDensity.text())  # 2700 #Set this value based on the above exercise original: 2500 only make sense if values have a huge GAP (good vs bad)
        reconstruction_error_threshold = float(self.lineEdit_threshReconstrErr.text())    #eg 0.002  Set this value based on the above exercise original example 0.004
        
        img  = Image.open(self._output_file)
        img = np.array(img.resize((128,128), Image.Resampling.LANCZOS)) # ! ANTIALIAS gibts nicht mehr'  128
        
        #preview on GUI
        imgx = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgx = QImage(imgx, imgx.shape[1],imgx.shape[0],imgx.strides[0],QImage.Format_RGB888) 
        pixmap = QtGui.QPixmap.fromImage(imgx)
        pixmap = pixmap.scaled(300, 300, QtCore.Qt.KeepAspectRatio)
        self.lbl_imgCNNinput.setPixmap(pixmap)   
                
       
        #plt.imshow(img)
        img = img / 255.
        img = img[np.newaxis, :,:,:]
        encoded_img = self.anomaly_model.predict([[img]]) 
        encoded_img = [np.reshape(img, (self.out_vector_shape)) for img in encoded_img] 
        density = self.kde.score_samples(encoded_img)[0] 
        density = round(density,0) 

        reconstruction = self.encodeDecode_model.predict([[img]])
        reconstruction_accurancy = self.encodeDecode_model.evaluate([reconstruction],[[img]], batch_size = 1)[0]
        reconstruction_accurancy = round(reconstruction_accurancy,4) 

        
       
        
        #mpimg.imsave("capFrame/myPredictImg.png", reconstruction )   
        
        #Due to issues with reconstruction-Image-format **NOTLÖSUNG** platte schreiben..Platte lesen!!??
        imgx = cv2.cvtColor(reconstruction[0], cv2.COLOR_RGBA2BGR)  # this the tric - convert plt-image to csv-image
        #https://stackoverflow.com/questions/53581609/using-imwrite-in-cv2-saves-a-black-image  
        cv2.imwrite("capFrame/myPredictImg.png", imgx * 255)   # SUPI
       
        #cv2.imshow('Image', imgx)
        imgx = cv2.imread("capFrame/myPredictImg.png")   # NOTLÖSUNG todo!!! ???   aber funtzt für den moment
        imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
        imgx = QImage(imgx, imgx.shape[1],imgx.shape[0],imgx.strides[0],QImage.Format_RGB888) 
        pixmap = QtGui.QPixmap.fromImage(imgx)
        pixmap = pixmap.scaled(300, 300, QtCore.Qt.KeepAspectRatio)
        self.lbl_imgPrediction.setPixmap(pixmap)   
        
        '''
        # reconstruction_accurancy statt reconstruction_error !
        zb wenn diese reconstruction_accurancy > dem schwellwert (2500) UND density < anomaly_reconstruction_error
        dann liegt keine anlomaly vor 
        '''
        isAnomaly = False   
        if  density < density_threshold and reconstruction_accurancy > reconstruction_error_threshold:
            isAnomaly = True        
            
        image = cv2.imread(self._output_file)      
        if  isAnomaly == True:
            print("The image is an anomaly") 
            img = toolz.draw_text(image, "ANOMALY", pos=(10, 10), text_color=(0, 0, 255) )
        else:
            print("The image is NOT an anomaly")       
            img = toolz.draw_text(image, "NO ANOMALY", pos=(10, 10), text_color=(0, 255, 0)   )
               
         
        self._statusBarTxt  =  "if " +  str(density) + " <  " +  str(density_threshold) + " and " +  str(reconstruction_accurancy) + " > " + str(reconstruction_error_threshold) + " : isAnomaly =" + str(isAnomaly)
        self.statusbar.showMessage (self._statusBarTxt)    
            
         #preview on GUI - Autoencoder RESULT
        imgx = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgx = QImage(imgx, imgx.shape[1],imgx.shape[0],imgx.strides[0],QImage.Format_RGB888) 
        pixmap = QtGui.QPixmap.fromImage(imgx)
        pixmap = pixmap.scaled(800, 800, QtCore.Qt.KeepAspectRatio)
        self.lbl_anomalyResult.setPixmap(pixmap)       
    
    '''        
    BT Read the docs - D:\ALL_DEVEL_VIP\DOC\Anaconda_PY_ObjectDetect_Tensor_ORANGE\How2_AnomalyDetection_EncoderDecoder_1.docx
    Received: _34#2765#34_
    Received: _39#0.1632#39_
    
    '''
    def getESP32_BluetoothPotiValues(self):
        try:
            # serial data received?  if so read byte and print it
            if self._SerialPort.in_waiting > 0:
                
                data = self._SerialPort.readline().decode('utf-8').strip()
                if "_34#" in data:
                    self._densityBT = data.split("_34#")[1].split("#34_")[0]
                    self.lineEdit_threshDensity_ESP32.setText(str(self._densityBT))                      
                if "_39#" in data:
                    self._reconstructErrBT = data.split("_39#")[1].split("#39_")[0]    
                    self.lineEdit_threshReconstrErr_ESP32.setText(str(self._reconstructErrBT))  
                    
                #print("Density = ",str(self._densityBT), "  reconErr = ", str(self._reconstructErrBT)  )    
        except:
            pass

#THIS sector is needed for stand alone mode 
        
def app():
    app = QtWidgets.QApplication(sys.argv)      
    win = window()
    win.show()    
    sys.exit(app.exec_())

app()   

