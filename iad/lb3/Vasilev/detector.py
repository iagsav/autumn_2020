import dlib
from keras.preprocessing import image
import numpy as np
import cv2
from deepface.commons import functions, realtime, distance as dst

def detect_face(img, detector_backend = 'dlib', grayscale = False, enforce_detection = True):
	
	#if functions.preproces_face is called directly, then face_detector global variable might not been initialized.

	detections = dlib.get_frontal_face_detector()(img, 1)
	
	if len(detections) > 0:
			
		for idx, d in enumerate(detections):
			left = d.left(); right = d.right()
			top = d.top(); bottom = d.bottom()
				
			detected_face = img[top:bottom, left:right]
			cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 3)
			return detected_face

def preprocess_face(img, target_size=(47, 55), grayscale = False, enforce_detection = True, detector_backend = 'dlib'):
	
	base_img = img.copy()
	
	img = detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection)
	if img is not None :
		#post-processing
		if grayscale == True:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
		img = cv2.resize(img, target_size)
		img_pixels = image.img_to_array(img)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		img_pixels /= 255 #normalize input in [0, 1]
	
		return img_pixels
