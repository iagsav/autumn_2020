import cv2
import numpy as np
import sklearn
from deepid_kremlev_model import create_model
from detecor_haar_kremlev import detectFace
from deepface.commons import distance as dst
from time import time
from sklearn.metrics.pairwise import cosine_distances


#video path
vPath = ('2.mp4')

#create video obj
video = cv2.VideoCapture(vPath) 

#Create_model
model = create_model()

###Weights load
model.load_weights("deepid_keras_weights.h5")

###read and detect source face to comare
img_r = cv2.imread("source.png")
arr = []
img1 = detectFace(img_r, (47, 55), arr)
source_img = model.predict(img1)

while(1<2):
    start = time()
    grab, frame = video.read()
    if not grab:
        raise Exception('Image not found!')

    arr = []
    #detect from new frame face
    img_2compare = detectFace(frame, (47, 55), arr)
    #try 
    if img_2compare is not None:
        ###Predict
        img2_representation = model.predict(img_2compare)

        ###Cosine
        '''
        cosine_distance_arr = dst.findCosineDistance(source_img, 
                                                    img2_representation)
        '''
        cosine = sklearn.metrics.pairwise.cosine_similarity(source_img,img2_representation)
        #print(cosine[0][0])
        ###CV2 text
        cv2.putText(frame, 'Me:{:.2f}'.format(cosine[0][0]), (arr[0], arr[1]),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

    ###FPS
    end = time()
    fps = 1 / (end - start)

    ###CV2 text
    cv2.putText(frame, 'fps:{:.2f}'.format(fps), (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    ###resize
    frame = cv2.resize(frame, (500,500), interpolation=cv2.INTER_AREA)
    
    #imshow
    cv2.imshow('Face Recognition Stream',frame)

    ###EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
