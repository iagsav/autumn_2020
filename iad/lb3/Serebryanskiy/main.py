from deepface import DeepFace
import cv2
import numpy as np

def photo_recognition():
    photo_path = 'C://Users//sawer//PycharmProjects//lab3//obama.jpg'
    img = cv2.imread(photo_path)
    img_copy = cv2.resize(img.copy(), (500, 500))
    cv2.imshow('Original', img_copy)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    detected_face = DeepFace.detectFace(img_bgr, detector_backend='ssd')
    cv2.imshow('Face', detected_face)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        print("process finished by user")

def photo_verification():
    photo_path = 'C://Users//sawer//PycharmProjects//lab3//obama.jpg'
    original_img = cv2.imread(photo_path)
    original_img_copy = cv2.resize(original_img.copy(), (500, 500))
    cv2.imshow('Original', original_img_copy)
    comp_path = 'C://Users//sawer//PycharmProjects//lab3//obama2.jpg'
    another_path = 'C://Users//sawer//PycharmProjects//lab3//photo.jpg'
    compared_img = cv2.imread(comp_path)
    another_img = cv2.imread(another_path)
    another_img = cv2.resize(another_img.copy(), (500, 500))
    cv2.imshow('another', another_img)
    cv2.imshow('Compared', compared_img)
    try:
        ver_result = DeepFace.verify(photo_path, comp_path, distance_metric='euclidean', model_name='Facenet',
                                     detector_backend='ssd')

        # вывод результата
        verified = bool(ver_result['verified'])
        if not verified:
            print("Разные люди")
        else:
            print("Один и тот же человек")
        key = cv2.waitKey(1) & 0xFF
    except cv2.error as e:
        print(e)

    try:
        ver_result = DeepFace.verify(photo_path, another_path, distance_metric='euclidean', model_name='Facenet',
                                     detector_backend='ssd')

        # вывод результата
        verified = bool(ver_result['verified'])
        if not verified:
            print("Разные люди")
        else:
            print("Один и тот же человек")
        key = cv2.waitKey(1) & 0xFF
    except cv2.error as e:
        print(e)



def video_recognition():
    cap = cv2.VideoCapture(0)
    while True:
        grab, original_img = cap.read()
        ssd_img = original_img.copy()
        ssd_bgr = cv2.cvtColor(ssd_img, cv2.COLOR_RGB2BGR)
        try:
            detected_face = DeepFace.detectFace(ssd_bgr, detector_backend='ssd')
            cv2.imshow("Face", detected_face)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Конец")
                break
        except ValueError as v:
            print(v)
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    print("Веб-камера или изображение?")
    print("1) Камера (только поиск лица)")
    print("2) Изображение")
    choice = int(input())
    # обработка видео с камеры
    if choice == 1:
            video_recognition()
    elif choice == 2:
        print("Поиск лица или верификация?")
        print("1) Поиск")
        print("2) Верификация")
        choice = int(input())
        if choice == 1:
            photo_recognition()
        elif choice == 2:
            photo_verification()
    else:
        raise ValueError("Unknown command!")
