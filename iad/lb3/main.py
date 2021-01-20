from deepface import DeepFace
import cv2
import numpy as np


def photo_recognition(path):
    # детектированиетолько с ипользованием DeepFace
    img = cv2.imread(path)
    img = cv2.resize(img, (500, 500))
    cv2.imshow('Photo', img)
    # распознавание лица
    detected_face = DeepFace.detectFace(img, detector_backend='opencv')
    cv2.imshow('Detected', detected_face)
    cv2.waitKey()




def video_verification(photo_path):
    original_img = cv2.imread(photo_path)
    original_img_copy = cv2.resize(original_img.copy(), (500, 500))
    cv2.imshow('Original', original_img_copy)
    # уменьшение размера фото, чтобы сеть могла его обработать
    original_img = cv2.resize(original_img.copy(), (160, 160))
    # увеличение размерности фото
    cap = cv2.VideoCapture(0)
    while True:
        grab, compared_img = cap.read()
        # уменьшение размера фото, чтобы сеть могла его обработать
        compared_img_copy = compared_img.copy()
        # увеличение размерности фото
        print("Сравниваем...")
        # сравнение исходного фото с кадром с камеры
        ver_result = DeepFace.verify(original_img, compared_img, distance_metric='euclidean', model_name='VGG-Face',
                                     detector_backend='opencv')
        cv2.imshow('Compared', compared_img)
        # вывод результата
        verified = bool(ver_result['verified'])
        if not verified:
            print("ИТОГ - Разные!")
        else:
            print("ИТОГ- Один и тот же")
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()



print("Привет, верификация по лицу или выберем фотографию с компьютера??")
print("1 - Верификация")
print("2 - Распознавание лица")
choice = int(input())
# обработка видео с камеры
if choice == 1:
    photo_path = "/home/ilsave/PycharmProjects/Lab3/IlsaveTest.jpg"
    video_verification(photo_path)
elif choice == 2:
    path = "/home/ilsave/PycharmProjects/Lab3/Ilsave.jpg"
    photo_recognition(path)
