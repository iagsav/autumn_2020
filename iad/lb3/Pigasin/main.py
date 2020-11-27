import cv2
import dlib
import numpy as np
from mtcnn import MTCNN
from scipy.spatial import distance


def resize_image(image, h=None, w=None):
    if h and not w:
        w = int(image.shape[1] * (h / image.shape[0]))
    elif not h and w:
        h = int(image.shape[0] * (w / image.shape[1]))
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)


def detect_faces(image, confidence=0.8):
    faces = []
    for detection in detector.detect_faces(image):
        if detection['confidence'] > confidence:
            faces.append((detection['box'], detection['keypoints']))
    return faces


def get_face_descriptor(image, box, show_face_chip=False):
    x, y, w, h = box
    shape = sp(image, dlib.rectangle(x, y, x + w, y + h))
    face_chip = dlib.get_face_chip(image, shape)
    if show_face_chip:
        image[0:150, 0:150] = face_chip
    return facerec.compute_face_descriptor(face_chip)


detector = MTCNN()
sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

photos = [cv2.imread(path) for path in ('1.jpg', '2.jpg')]
photo_boxes = [detect_faces(photo)[0][0] for photo in photos]
photo_descriptors = [get_face_descriptor(photos[i], photo_boxes[i]) for i in range(len(photos))]
resized_photos = [resize_image(photo, h=300) for photo in photos]

photo_board = np.concatenate(resized_photos, axis=1)

cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    res_board = photo_board.copy()
    for box, keypoints in detect_faces(frame):
        x, y, w, h = box
        face_descriptor = get_face_descriptor(frame, (x, y, w, h), show_face_chip=True)

        distances = [
            distance.euclidean(face_descriptor, photo_descriptor)
            for photo_descriptor in photo_descriptors
        ]

        cv2.putText(
            res_board,
            f'{distances[0]:.2f}',
            (resized_photos[0].shape[1] - 100, res_board.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            res_board,
            f'{distances[1]:.2f}',
            (res_board.shape[1] - 100, res_board.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for key in keypoints.keys():
            cv2.circle(frame, keypoints[key], 3, (255, 0, 255), -1)

    cv2.imshow('cam', frame)
    cv2.imshow('photo_board', res_board)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
