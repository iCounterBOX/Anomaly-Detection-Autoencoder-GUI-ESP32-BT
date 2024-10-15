# https://youtu.be/q_tpFGHiRgg
"""
Detecting anomaly images using AutoEncoders. 
(Sorting an entire image as either normal or anomaly)

Here, we use both the reconstruction error and also the kernel density estimation
based on the vectors in the latent space. We will consider the bottleneck layer output
from our autoencoder as the latent space. 

Training Material:

Data from: https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html

04.08.24: Findings from 2Day:
    Der erste wurf - in einem dataframe density und recoErr im run zu ermitteln (median ) fallen gelassen.
    2. ansatz density, reconErr in calc_density_and_recon_error(batch_images) über ALLE batches ist OK..aber der medianWelt
    kann SO auch nicht als konstante verwendet werden!
    Im Moment ist der theshold "on the fly" ermittelt...NUR noch der value der GOODvalues während der webCam werden betrachtet.
    INFO: die DENSITY ist "NearBy" zwischen Good & Bad...Haben sie hier in diesem Experiment ( Pumpe  + bad käbele ) heraus genommen. 
    NUR noch threshold/ reconErr !!!
    Diese werte werden +/- dann dals threshold eingesetzt.
    
    ToDo:
        pyQt5 mit diesen threshold sliders,,im run können wir dann auf GOOD eichen!?
    
"""

import cv2  # install macht probleme ...siehe doc
import sys
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KernelDensity

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import random


from ad_class import ad_class

cocr = ad_class()   # class ocr

#.. some kind of global ..........

_camNr = 1


#Size of our input images
SIZE =  128


#############################################################################
#Define generators for training, validation and also anomaly data.

batch_size = 64
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    'trainImages/errorFree_train/',
    target_size=(SIZE, SIZE),
    batch_size=batch_size,
    class_mode='input'
    )

validation_generator = datagen.flow_from_directory(
    'trainImages/errorFree_test/',
    target_size=(SIZE, SIZE),
    batch_size=batch_size,
    class_mode='input'
    )

anomaly_generator = datagen.flow_from_directory(
    'trainImages/defect/',
    target_size=(SIZE, SIZE),
    batch_size=batch_size,
    class_mode='input'
    )


'''
H i e r   g e h t   d i e   A n o m a l y - D e t e c t i o n   l o s 

'''
# WE LOAD THE SAVED and TRAINED ANOMALY - MODEL - gebaut mit: 1_anomaly_make_h5_Model.py

encodeDecode_model = load_model('model_encodeDecode.h5')
anomaly_model = load_model('model_anomalyFinal.h5')


'''

# Calculate KDE using sklearn - Kernel Density Estimation
we use Kernel-Density-Esimation (KDE) to calculate the likelihood of an image belonging to the good class

'''

#Get encoded output of input images = Latent space / Model.predict_generator` is deprecated and will be removed in a future version. Please use `Model.predict`,
encoded_images = anomaly_model.predict(train_generator)

# Flatten the encoder output because KDE from sklearn takes 1D vectors as input
encoder_output_shape = anomaly_model.output_shape #Here, we have 16x16x16
out_vector_shape = encoder_output_shape[1]*encoder_output_shape[2]*encoder_output_shape[3]

encoded_images_vector = [np.reshape(img, (out_vector_shape)) for img in encoded_images]

#Fit KDE to the image latent data -  D A U E R T   -   T I M E   c o n s u m i n g 
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(encoded_images_vector)  # <<<<<<< KDE

#Calculate density and reconstruction error to find their means values for
#good and anomaly images. 
#We use these mean and sigma to set thresholds. 
def calc_density_and_recon_error(batch_images):
    
    density_list=[]
    recon_error_list=[]
    for im in range(0, batch_images.shape[0]-1):
        
        img  = batch_images[im]
        img = img[np.newaxis, :,:,:]
        encoded_img = anomaly_model.predict([[img]]) # Create a compressed version of the image using the encoder
        encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img] # Flatten the compressed image
        density = kde.score_samples(encoded_img)[0] # get a density score for the new image
        reconstruction = encodeDecode_model.predict([[img]])
        
        reconstruction_error = encodeDecode_model.evaluate([reconstruction],[[img]], batch_size = 1)[0]
        density_list.append(density)
        recon_error_list.append(reconstruction_error)
        
    average_density = np.mean(np.array(density_list))  
    stdev_density = np.std(np.array(density_list)) 
    
    average_recon_error = np.mean(np.array(recon_error_list))  
    stdev_recon_error = np.std(np.array(recon_error_list)) 
    
    return average_density, stdev_density, average_recon_error, stdev_recon_error

#Get average and std dev. of density and recon. error for uninfected and anomaly (parasited) images. 
#For this let us generate a batch of images for each. 

dfColumnNames = [
            'kde_density',  # measured during the anomaly detection
            'reconstruction_accurancy'
            ]
_df1 = pd.DataFrame(columns = dfColumnNames)


# !!! ONLY to get some parameter for the NEXT check_anomaly fct: ( OPTIONAL )
# we write the density and reconErr from ALL batches in a dataframe ..finally MEDIAN to get VALUES
# in some case the density value between err and NoErrObject is nearBy...so the recoAccurance with OR ist better i guess
#...depends on the heavyaness of the Failure
'''
def get_median4density_reconstructionErr():
    img_num = 0
    while img_num <= train_generator.batch_index:   #gets each generated batch of size batch_size
        train_batch = train_generator.next()[0]    
        values_uninfected = calc_density_and_recon_error(train_batch)
        print("values_uninfected / Density: "  +  str(values_uninfected[0] )   )
        print("values_uninfected / reconErr: "  +  str(values_uninfected[2] )   )
        newRow = {   
            'kde_density': values_uninfected[0], 
            'reconstruction_accurancy': values_uninfected[2] 
            }
        _df1.loc[len(_df1)] = newRow
        print(_df1) # OHNE shift
        img_num = img_num + 1
   
get_median4density_reconstructionErr()
_density_median = _df1['kde_density'].median()
_reconstruction_accurancy_median = _df1['reconstruction_accurancy'].median()    
print ("density_median                 :", str(_density_median ) )
print ("reconstruction_accurancy_median:", str(_reconstruction_accurancy_median ) )
'''



# !!! ONLY to get some parameter for the NEXT check_anomaly fct: ( OPTIONAL )
'''
train_batch = train_generator.next()[0]
anomaly_batch = anomaly_generator.next()[0]
values_uninfected = calc_density_and_recon_error(train_batch)
values_anomaly = calc_density_and_recon_error(anomaly_batch)
'''



#Now, input unknown images and sort as Good or Anomaly
def check_anomaly(img_path):
    density_threshold = 2700 #Set this value based on the above exercise original: 2500 only make sense if values have a huge GAP (good vs bad)
    reconstruction_error_threshold = 0.002 # Set this value based on the above exercise original example 0.004
    print(img_path)
    img  = Image.open(img_path)
    img = np.array(img.resize((128,128), Image.Resampling.LANCZOS)) # ! ANTIALIAS gibts nicht mehr'  128
    
    cocr.showInMovedWindow(  "128x128 scan image", img, 50, 50)
    plt.imshow(img)
    img = img / 255.
    img = img[np.newaxis, :,:,:]
    encoded_img = anomaly_model.predict([[img]]) 
    encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img] 
    density = kde.score_samples(encoded_img)[0] 

    reconstruction = encodeDecode_model.predict([[img]])
    reconstruction_accurancy = encodeDecode_model.evaluate([reconstruction],[[img]], batch_size = 1)[0]
    
    print("density: " + str(round(density)) +  " reconstruction_accurancyc: " + str( round(reconstruction_accurancy,4)))
    #mpimg.imsave("capFrame/myPredictImg.png", reconstruction )   
    
    csvImage = cv2.cvtColor(reconstruction[0], cv2.COLOR_RGBA2BGR)  # this the tric - convert plt-image to csv-image
    cocr.showInMovedWindow(  "prediction", csvImage, 900, 50)

    
    '''
    # reconstruction_accurancy statt reconstruction_error !
    zb wenn diese reconstruction_accurancy > dem schwellwert (2500) UND density < anomaly_reconstruction_error
    dann liegt keine anlomaly vor 
    '''
           
    image = cv2.imread(img_path)
    image = cv2.resize(image, (640, 480) )  # damit es auf den nb screen passt ..nearbyIssue..: density < density_threshold and
    ##if  density < density_threshold or reconstruction_accurancy > reconstruction_error_threshold:
    if  density < density_threshold and reconstruction_accurancy > reconstruction_error_threshold:
        print("The image is an anomaly") 
        cocr.msgWithTextOnImage(  "ANOMALY_CHECK", "ANOMALY !!", image, 200, 50 , (255,0,0))
        
    else:
        print("The image is NOT an anomaly")
        cocr.msgWithTextOnImage(  "ANOMALY_CHECK","NO ANOMALY", image, 200, 50, (34,139,34))
        
        
#Load a couple of test images and verify whether they are reported as anomalies.
import glob
para_file_paths = glob.glob('trainImages/defect/images/*')
uninfected_file_paths = glob.glob('trainImages/errorFree_train/images/*')

#Anomaly image verification
num=random.randint(0,len(para_file_paths)-1)
check_anomaly(para_file_paths[num])

#Good/normal image verification
num=random.randint(0,len(uninfected_file_paths)-1)
check_anomaly(uninfected_file_paths[num])

#beliebiges file:
check_anomaly('capFrame/myFrame.jpg')
    


# W E B C A M
# w e b c a m   -   L O O P 

cap = cv2.VideoCapture(_camNr ,cv2.CAP_DSHOW )
cap.set(3,1920) # MUSS rein sonst fällt er auf 640x480 zurück / usb-microscope: 1280 x 720 / usb-webcam 1920 x 1080
cap.set(4,1080)
cap.set(5,30)


# set the start time
start_time = time.time()

# loop through the frames
print("looping through frames")
while True:
    # capture a frame
    ret, frame = cap.read()

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)

    # define the output file name and format
    output_file = 'capFrame/myFrame.jpg'    

    # check if the frame is captured successfully
    if not ret:
        break

    # calculate the elapsed time
    elapsed_time = time.time() - start_time

    # check if 1 second has passed
    if elapsed_time >= 0.5:
        #print("1 second passed")

            
        # save the frame as an image file
        is_ok = cv2.imwrite(output_file, frame)

        # reset the start time
        start_time = time.time()
        #print("time reset")
        
        # display the frame
        #cv2.imshow('frame', frame)
        
        if is_ok:
            check_anomaly(output_file)

    key = cv2.waitKey(1)
    if key == ord('q'):        
        break
    if key == ord('p'):
        print('press c to continue')        
        cv2.waitKey(-1) #wait until any key is pressed
        if key == ord('c'):          
            print('c pressed')

# release the camera and close all windows
cap.release()

cv2.destroyAllWindows()




sys.exit()



