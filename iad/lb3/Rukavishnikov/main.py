from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from deepface.basemodels import Facenet
import pandas as pd
import dlib
from skimage import io


def show_images(img1, img2):
    img1 = cv2.imread(imgs[0])
    img2 = cv2.imread(imgs[1])
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_face(img):
    orig = cv2.imread(img)
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    detected_face = DeepFace.detectFace(img, detector_backend = 'mtcnn')
    detector = dlib.get_frontal_face_detector()

    cv2.imshow("Original image", orig)
    cv2.imshow("Detected face", detected_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_landmarks(img):
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    detected_face = DeepFace.detectFace(img, detector_backend = 'mtcnn')
    detector = dlib.get_frontal_face_detector()

    img = io.imread(img)

    win1 = dlib.image_window()
    win1.clear_overlay()
    win1.set_image(img)

    dets = detector(img, 1)

    print(dets)

    for k,d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        shape = sp(img, d)
        win1.clear_overlay()
        win1.add_overlay(d)
        win1.add_overlay(shape)
        win1.wait_until_closed()

    face_desc = facerec.compute_face_descriptor(img, shape)
     

def verify_images(img1, img2, results):
    obj = DeepFace.verify(img1, img2, distance_metric='euclidean', model_name='Facenet', detector_backend = 'mtcnn')
    results.append(obj)


def get_video_verification(path, img, results, frames):
    img = cv2.imread(img)
    img = cv2.resize(img, (400, 400))
    cv2.imshow("Photo", img)
    cap = cv2.VideoCapture(path)
    count_frames = 0
    while True:
        if count_frames == (frames + 1):
            return        
        ret, frame = cap.read()
        frame = cv2.resize(frame, (400, 400))
        cv2.putText(frame, ("frame " + str(count_frames + 1)), (10, 30), fontFace = 1, fontScale = 2, color = (0, 0, 255), thickness = 2)
        cv2.imshow("camera", frame)
        if not count_frames == 0:
            verify_images(frame, img, results)
        count_frames += 1
        if cv2.waitKey(10) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def get_verification_results(results):
    count = 1
    for el in results:
        print(str(count)+") " + str(el["verified"]))
        count += 1


if __name__ == "__main__":
    imgs = ["1.jpg", "2.jpg", "head1.jpg", "head2.jpg"]
    video = 'video.mov'
    results = []
    frames = 3 # Количество кадров


    while(True):
        print("[1] - Face detection")
        print("[2] - Draw landmarks (dlib)")
        print("[3] - Video verification")
        print("[4] - Exit")
        print("[5] - Images verification")
        
        ch = int(input())
    
        if (ch==1):
            detect_face(imgs[0])
        elif (ch==2): 
            show_landmarks(imgs[0])
        elif (ch==3):
            get_video_verification(video, imgs[0], results, frames)
            get_verification_results(results)
        elif (ch==4):
            exit()
        elif (ch==5):
            verify_images(imgs[0], imgs[3], results)
            get_verification_results(results)