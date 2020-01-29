"""
 __  __           _     __        __              _             
|  \/  | __ _ ___| | __ \ \      / /__  __ _ _ __(_)_ __   __ _ 
| |\/| |/ _` / __| |/ /  \ \ /\ / / _ \/ _` | '__| | '_ \ / _` |
| |  | | (_| \__ \   <    \ V  V /  __/ (_| | |  | | | | | (_| |
|_|  |_|\__,_|___/_|\_\    \_/\_/ \___|\__,_|_|  |_|_| |_|\__, |
                                                          |___/ 

@author: Jonathan Wang
@coding: utf-8
@environment: Manjaro 18.1.5 Juhraya
@date: 29th Jan., 2020

"""
import os
import dlib
import random
import cv2 as cv
import numpy as np

def detect_mouth(img):

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('src/models/shape_predictor_68_face_landmarks.dat')
    faces = detector(img_gray, 0)

    for k, d in enumerate(faces):
        x = []
        y = []
        height = d.bottom() - d.top()
        width = d.right() - d.left()
        shape = predictor(img_gray, d)

        # get the mouth part
        for i in range(48, 68):
            x.append(shape.part(i).x)
            y.append(shape.part(i).y)

        y_max = (int)(max(y) + height / 3)
        y_min = (int)(min(y) - height / 3)
        x_max = (int)(max(x) + width / 3)
        x_min = (int)(min(x) - width / 3)
        size = ((x_max-x_min),(y_max-y_min))

    return x_min, x_max, y_min, y_max, size


def detect_eye(img):

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('src/models/shape_predictor_68_face_landmarks.dat')
    faces = detector(img_gray, 0)

    for k, d in enumerate(faces):
        x = []
        y = []
        height = d.bottom() - d.top()
        width = d.right() - d.left()
        shape = predictor(img_gray, d)

        for i in range(36, 48):
            x.append(shape.part(i).x)
            y.append(shape.part(i).y)

        y_max = (int)(max(y) + height / 3)
        y_min = (int)(min(y) - height / 3)
        x_max = (int)(max(x) + width / 3)
        x_min = (int)(min(x) - width / 3)
        size = ((x_max-x_min),(y_max-y_min))

    return x_min, x_max, y_min, y_max, size


def wear_item(mask):

    print("Processing...")

    if not mask:
        x_min, x_max, y_min, y_max, size = detect_eye(img)
        item_img = cv.imread('src/imgs/glasses.png', cv.IMREAD_UNCHANGED)
        #  cv.imshow("Glasses", item_img)
    else:
        x_min, x_max, y_min, y_max, size = detect_mouth(img)
        which = random.randint(0, 4)
        item_name = 'src/imgs/mask' + str(which) + '.png'
        item_img = cv.imread(item_name, cv.IMREAD_UNCHANGED)
        #  cv.imshow("Mask", item_img)

    item_img = cv.resize(item_img, size)
    alpha_channel = item_img[:, :, 3]
    _, mask = cv.threshold(alpha_channel, 220, 255, cv.THRESH_BINARY)
    color = item_img[:, :, :3]
    item_img = cv.bitwise_not(cv.bitwise_not(color, mask=mask))

    rows, cols, channels = item_img.shape
    roi = img[y_min: y_min + rows, x_min:x_min + cols]
    img_gray = cv.cvtColor(item_img, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img_gray, 254, 255, cv.THRESH_BINARY)
    mask = np.uint8(mask)
    mask_inv = cv.bitwise_not(mask)
    img_bg = cv.bitwise_and(roi, roi, mask=mask)
    item_img_fg = cv.bitwise_and(item_img, item_img, mask=mask_inv)
    dst = cv.add(img_bg, item_img_fg)
    img[y_min: y_min + rows, x_min:x_min + cols] = dst


if __name__ == '__main__':
    raw_img = cv.imread('src/imgs/faces/riho2.jpg')
    img = raw_img.copy()

    x = input('Wear Glasses? [Y/n]')
    if x == 'Y' or x == 'y':
        wear_item(False)
        print("OK!")

    x = input('Wear a mask? [Y/n]')
    if x == 'Y' or x == 'y':
        wear_item(True)
        print("OK!")

    cv.imshow('Raw', raw_img)
    cv.imshow('Result', img)

    cv.waitKey(0)
    cv.destroyAllWindows()

