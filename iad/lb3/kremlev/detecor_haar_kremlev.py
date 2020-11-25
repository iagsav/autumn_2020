import cv2
import numpy as np
from keras.preprocessing import image


def detectFace(img, target_size,arr):
    #img = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier("C:/Users/kreml/Documents/5sem/IAD/3lab/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=6,minSize=(20, 20))
    
    if (len(faces)>0):
        height, _ = img.shape[:2]
        x,y,w,h = faces[0]
        if y < (height / 2): 
            arr.append(x)
            arr.append(y)
            detected_face = img[int(y):int(y+h), int(x):int(x+w)]
            detected_face = cv2.resize(detected_face, target_size)
            cv2.rectangle(img, (x, y), (x + w, y + h +5), (0, 0, 255), 3)
            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)

            if True:
                #normalize input in [0, 1]
                img_pixels /= 255 
            else:
                #normalize input in [-1, +1]
                img_pixels /= 127.5
                img_pixels -= 1

            return img_pixels