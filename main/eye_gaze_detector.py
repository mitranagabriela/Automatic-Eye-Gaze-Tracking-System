# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:52:47 2019

@author: Gabriela
"""

from localbinarypatterns import LocalBinaryPatterns
from sample import Sample
from sklearn.svm import LinearSVC
from collections import OrderedDict
import pickle

import cv2
import sys
import os
import numpy
import imutils
import dlib

data_set = []
labels_set = []
eye_coords_set = [] 
images_set = [] 

list_of_eye_coords = []
list_of_hists = [] 
list_of_images = [] 

#initialize the classifier
model = LinearSVC(C=100.0, random_state=42)

#initialize the descriptor model 
desc = LocalBinaryPatterns(24, 8)

#var_initial is 0 only when training
var_initial = 1

# initialize haar cascade eye and face detector
face_cascade = cv2.CascadeClassifier('D:/Thesis/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('D:/Thesis/haarcascade_eye.xml')

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Thesis/shape_predictor_68_face_landmarks.dat')

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("right_eye", (36, 42)),
	("left_eye", (42, 48))
])

def resize_image(image, scale_percent):
  
  width = int(image.shape[1] *  scale_percent / 100)
  height = int(image.shape[0] *  scale_percent / 100)
  dim = (width, height)
  resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  return resized

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = numpy.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

###################################################
############FOLDER PROCESSING######################
################################################### 
    
#returns all the immediate directories from a_dir
def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name).replace('\\', '/') for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]    
    
#returns a list containg all the images from a directory    
def load_images_from_folder(folder):
  images = []
  path, dirs, files = next(os.walk(folder))
  for filename in os.listdir(folder):      
    if any([filename.endswith(x) for x in ['.jpeg', '.jpg']]):
      img = cv2.imread(os.path.join(folder, filename))
      if img is not None:
        images.append(img)
  return images  

###################################################
############END OF FOLDER PROCESSING###############
################################################### 

###################################################
###########DETECTION METHODS#######################
###################################################   
def detect_faces_and_eyes_with_haar(image): 
    
  list_of_hists[:] = []
  list_of_eye_coords[:] = []
  
  if(not(numpy.array_equal(numpy.array(image.shape), numpy.array([480, 640, 3])))):
    image = resize_image(image, 41)
    
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
  
  #detect faces
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x, y, w, h) in faces:
    
    #draws a rectangle for the detected face
    #cv2.rectangle(image, (x, y),(x+w, y+h), (255, 0, 0), 2)
    
  	 #delimits the detected area for the face	
    face_roi_gray = gray[y:y+h, x:x+w]
    #face_roi_color = image[y:y+190, x:x+210]        
    			
    #haar detects in some images more than two 'eyes' so
    #I have to limit the number to 2 for better precision
    nr_of_eyes = 0
    
    #detect eyes in the face area
    eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 3)      
    for (ex, ey, ew, eh) in eyes:
      
      #nr_of_eyes->assures me that there weren't more than 2 eyes detected
      nr_of_eyes = nr_of_eyes + 1
      
      #draws a rectangle for the detected eye   
      #cv2.rectangle(face_roi_color, (ex, ey), (ex+50, ey+50), (0, 255, 0), 2)
      
      #delimit the eye area       
      eye_roi_gray = gray[ey:ey+eh, ex:ex+ew]      
      
      #describe the area of the eyes
      hist = desc.describe(eye_roi_gray)
      list_of_hists.append(hist)
      list_of_eye_coords.append(numpy.array([x+ex, y+ey, ew, eh]))
      
#      cv2.imshow('img',image)
#      if cv2.waitKey(0) & 0xFF == ord('q'):
#        cv2.destroyAllWindows()
#        sys.exit()
#      elif cv2.waitKey(0) & 0xFF == ord('n'):
#        cv2.destroyAllWindows()
#        break
#      else:
#        cv2.destroyAllWindows()  
#        
      if nr_of_eyes == 2: break   
    
  return Sample(image, list_of_hists, list_of_eye_coords) 

def detect_faces_and_eyes_with_dlib(image):  
    
  list_of_hists[:] = []
  list_of_eye_coords[:] = []  
  
  if(not(numpy.array_equal(numpy.array(image.shape), numpy.array([480, 640, 3])))):
    image = resize_image(image, 41)
    
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  #detects faces
  rects = detector(gray, 1)
  # loop over the face detections
  for (i, rect) in enumerate(rects):
    
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
  
    # loop over the face parts individually
    for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
      
      # clone the original image so we can draw on it
      #clone = image.copy()
      #cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
  
      # loop over the subset of facial landmarks, drawing the
      # specific face part
      #for (x, y) in shape[i:j]:
        #cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
        
      #extract the ROI of the face region as a separate image
      (x, y, w, h) = cv2.boundingRect(numpy.array([shape[i:j]]))
      roi = image[y:y + h, x:x + w]
      roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
      roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
      
      hist = desc.describe(roi_gray)        
      list_of_hists.append(hist)
      list_of_eye_coords.append(numpy.array([x, y, w, h]))
      
#      cv2.imshow('img',image)
#      if cv2.waitKey(0) & 0xFF == ord('q'):
#        cv2.destroyAllWindows()
#        sys.exit()
#      elif cv2.waitKey(0) & 0xFF == ord('n'):
#        cv2.destroyAllWindows()
#        break
#      else:
#        cv2.destroyAllWindows()  
        
  return Sample(image, list_of_hists, list_of_eye_coords) 
###################################################
##########END OF DETECTION METHODS#################
################################################### 

###################################################
############IMAGES PROCESSING######################
###################################################   

def process_images_from_folder_var1(folder, begin_value, end_value, var, loaded_model):  
  data_set[:] = []
  labels_set[:] = []
  eye_coords_set[:] = [] 
  images_set[:] = []    
  
  images = load_images_from_folder(folder)    
  #75% from the total nr of images is used for training
  #25% from the total nr of images is used for testing
  starting_num = int((begin_value * len(images))/100)
  target = int((end_value * len(images))/100)
#  print(target-starting_num)
  for cont in range(starting_num, target):    
    #var determines which method of detection to be used
    #is set depending on the input written by the user
    #var == 0 for 'haar'
    #var == 1 for 'dlib'
    if(var == 0):
      sample = detect_faces_and_eyes_with_haar(images[cont])
    else: 
      sample = detect_faces_and_eyes_with_dlib(images[cont]) 
    if len(sample.eye_descript)!= 0:
      for i in range (0, len(sample.eye_descript)):      
        data_set.append(sample.eye_descript[i])
        labels_set.append(folder.split("/")[-1].split(".")[-1]) 
        eye_coords_set.append(sample.eye_coord[i])
        images_set.append(sample.image)         
  return data_set, labels_set, eye_coords_set, images_set
   
def process_images_from_folder_var2(folder, begin_value, end_value, var, loaded_model):  
  
  data_set[:] = []
  labels_set[:] = []
  eye_coords_set[:] = [] 
  images_set[:] = []    
      
  images = load_images_from_folder(folder)  
  
  #75% from the total nr of images is used for training
  #25% from the total nr of images is used for testing
  starting_num = int((begin_value * len(images))/100)
  target = int((end_value * len(images))/100)
#  print(target-starting_num)
  for cont in range(starting_num, target):    
    #var determines which method of detection to be used
    #is set depending on the input written by the user
    #var == 0 for 'haar'
    #var == 1 for 'dlib'
    if(var == 2):
      sample = detect_faces_and_eyes_with_haar(images[cont])
    else: 
      sample = detect_faces_and_eyes_with_dlib(images[cont]) 
    
    #the arrays are initialized with zero because you can't append 
    #to something undefined
    hists = numpy.array([0])
    eye_coords = numpy.array([0])
    
   #concatenate data from the 2 eyes
    for i in range (0, len(sample.eye_descript)):
      hists = numpy.concatenate((hists, sample.eye_descript[i]), axis=None) 
      eye_coords = numpy.concatenate((eye_coords, sample.eye_coord[i]), axis=None)
      
    #delete the first zero introduced  
    hists = numpy.delete(hists, [0])
    eye_coords = numpy.delete(eye_coords, [0])  
    
    #in case one single eye is detected double the information
    #obtained from one eye
    if len(sample.eye_descript) == 1:
      hists = numpy.concatenate((hists, hists), axis=None) 
      eye_coords = numpy.concatenate((eye_coords, eye_coords), axis=None)      
    
    if len(hists)!= 0:
      data_set.append(hists)
      labels_set.append(folder.split("/")[-1].split(".")[-1])
      eye_coords_set.append(eye_coords)
      images_set.append(sample.image)       
        
  return data_set, labels_set, eye_coords_set, images_set 

###################################################
##########END OF IMAGES PROCESSING#################
################################################### 

###################################################
############HANDLING USER'S INPUT##################
###################################################     
  
def choose_processing_and_detection():
  txt_p = input("What processing method do you want to use: v1/v2?")
  if(txt_p == 'v1'):
    txt = input("What detection method do you want to use: haar/dlib?")
    if(txt == 'haar'): 
      var = 0
      filename = 'finalized_model_haar_v1.sav'
    else:
      var = 1 
      filename = 'finalized_model_dlib_v1.sav'
  else:
    txt = input("What detection method do you want to use: haar/dlib?")
    if(txt == 'haar'): 
      var = 2
      filename = 'finalized_model_haar_v2.sav'
    else:
      var = 3 
      filename = 'finalized_model_dlib_v2.sav'
  return var, filename 

###################################################
#########END OF HANDLING USER'S INPUT##############
###################################################     
   
###################################################
############TRAINING PART##########################
################################################### 
def train_model(var_initial, var, filename, folders): 
  final_training_data_set = [0]
  final_training_labels_set = [0]
  for folder in folders:   
    if(var == 0) or (var == 1): 
      training_data_set, training_labels_set, _, _ = process_images_from_folder_var1(folder, 0, 75, var, 0)       
    else:
      training_data_set, training_labels_set, _, _ = process_images_from_folder_var2(folder, 0, 75, var, 0)   
    final_training_data_set.extend(training_data_set) 
    final_training_labels_set.extend(training_labels_set)
  del final_training_data_set[0]
  del final_training_labels_set[0]
#  print(final_training_data_set)
#  print(final_training_labels_set)
  model.fit(final_training_data_set, final_training_labels_set)      
  pickle.dump(model, open(filename, 'wb'))     

###################################################
############END OF TRAINING PART###################
################################################### 

###################################################
#############TESTING PART##########################
###################################################
def test_images(loaded_model, folders):
  for folder in folders:  
    testing_data_set = []
    testing_eye_coords_set = []
    testing_images_set = []
    correctly_predicted = 0
    incorrectly_predicted = 0
    if(var == 0) or (var == 1): 
      testing_data_set, _, testing_eye_coords_set, testing_images_set = process_images_from_folder_var1(folder, 75, 100, var, loaded_model)
      for i in range (len(testing_data_set)):
        prediction = loaded_model.predict(testing_data_set[i].reshape(1, -1))
        clone = testing_images_set[i].copy()
        cv2.putText(clone, prediction[0], (100, 300), cv2.FONT_HERSHEY_SIMPLEX,	1.0, (0, 0, 255), 3)
       
        if(prediction[0] == folder.split("/")[-1].split(".")[-1] ):
          correctly_predicted = correctly_predicted + 1
        else:
          incorrectly_predicted = incorrectly_predicted + 1
          
        rectangle_coords = testing_eye_coords_set[i]
        x = rectangle_coords[0]
        y = rectangle_coords[1]
        w = rectangle_coords[2]
        h = rectangle_coords[3]
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('img', clone)
        if cv2.waitKey(1000) & 0xFF == ord('n'):
          cv2.destroyAllWindows()
          break
        elif cv2.waitKey(1000) & 0xFF == ord('q'): 
          cv2.destroyAllWindows()
          sys.exit()
        else:
          cv2.destroyAllWindows()
          
        print(folder.split("/")[-1].split(".")[-1]) 
        print("\n")         
#      print(correctly_predicted)
#      print(incorrectly_predicted)
#      print("*******")
    else:
      testing_data_set, _, testing_eye_coords_set, testing_images_set = process_images_from_folder_var2(folder, 75, 100, var, loaded_model)
      for i in range (len(testing_data_set)):
        prediction = loaded_model.predict(testing_data_set[i].reshape(1, -1))
        clone = testing_images_set[i].copy()
        cv2.putText(clone, prediction[0], (100, 300), cv2.FONT_HERSHEY_SIMPLEX,	1.0, (0, 0, 255), 3)
        
        if(prediction[0] == folder.split("/")[-1].split(".")[-1]):
          correctly_predicted = correctly_predicted + 1
        else:
          incorrectly_predicted = incorrectly_predicted + 1
          
        rectangle_coords = testing_eye_coords_set[i]
        x1 = rectangle_coords[0]
        y1 = rectangle_coords[1]
        w1 = rectangle_coords[2]
        h1 = rectangle_coords[3]
        x2 = rectangle_coords[4]
        y2 = rectangle_coords[5]
        w2 = rectangle_coords[6]
        h2 = rectangle_coords[7]
        cv2.rectangle(clone, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        cv2.rectangle(clone, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
      
        cv2.imshow('img', clone)
        if cv2.waitKey(0) & 0xFF == ord('n'):
          cv2.destroyAllWindows()
          break
        elif cv2.waitKey(0) & 0xFF == ord('q'): 
          cv2.destroyAllWindows()
          sys.exit()
        else:
          cv2.destroyAllWindows()
          
        print(folder.split("/")[-1].split(".")[-1])
        print("\n") 
#      print(correctly_predicted)
#      print(incorrectly_predicted)
#      print("*******")

###################################################
#############END OF TESTING PART###################
###################################################

folders = get_immediate_subdirectories('C:/Users/Gabriela/Desktop/Eye_chimeraToPublish')
var, filename = choose_processing_and_detection()
print("Started training")      
if(var_initial == 0):
  train_model(var_initial, var, filename, folders)
  var_initial = 1
print("Ended training")  
print("\n")   
print("Started testing")     
print("\n")    
loaded_model = pickle.load(open(filename, 'rb'))    
test_images(loaded_model, folders)
print("Ended testing") 