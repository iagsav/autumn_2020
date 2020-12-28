import cv2
import numpy as np
import sklearn
from deepface.commons import distance as dst
from time import time
import dlib
from sklearn.metrics.pairwise import cosine_distances
from deepid_model import create_model
import detector
from deepface.commons import functions, realtime, distance as dst

# Прописываем путь к видео
videoPath = '1234.mp4'

# Захватываем видео
video = cv2.VideoCapture(0) 

# Создаем модель
model = create_model()

# Загружаем веса
model.load_weights("deepid_keras_weights.h5")

#read and detect source face to comare

img_r = cv2.imread("sour.jpg")
img1 = detector.preprocess_face(img=img_r
                    , target_size=(47, 55)
                    , detector_backend = 'dlib')
source_img = model.predict(img1)

while(True):
    start = time()
    grab, frame = video.read()
    if not grab:
        raise Exception('Image not found!')

    #detect from new frame face
    img_2compare = detector.preprocess_face(img=frame
                    , target_size=(47, 55)
                    , detector_backend = 'dlib')
    #try 
    if img_2compare is not None:
        ###Predict
        img2_representation = model.predict(img_2compare)

        ###Cosine
        cosine = sklearn.metrics.pairwise.cosine_similarity(source_img,img2_representation)
        ###CV2 text
        cv2.putText(frame, 'Kirill:{:.2f}'.format(cosine[0][0]), (470, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

    #FPS
    end = time()
    fps = 1 / (end - start)

    #CV2 text
    cv2.putText(frame, 'fps:{:.2f}'.format(fps), (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    #resize
    frame = cv2.resize(frame, (500,500), interpolation=cv2.INTER_AREA)
    
    #imshow
    cv2.imshow('Stream',frame)

    #EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()