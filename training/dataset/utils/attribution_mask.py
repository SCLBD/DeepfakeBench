

import cv2
import math
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def remove_mouth(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[3:5]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    return line

def remove_eyes(image, landmarks, opt='b'):
    ##l: left eye; r: right eye, b: both eye
    if opt == 'l':
        (x1, y1), (x2, y2) = landmarks[36],landmarks[39]
    elif opt == 'r':
        (x1, y1), (x2, y2) = landmarks[42],landmarks[46]
    elif opt == 'b':
        (x1, y1), (x2, y2) = landmarks[36],landmarks[46]
    else:
        print('wrong region')
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(np.array(mask, dtype=np.uint8), (int(x1), int(y1)), (int(x2), int(y2)), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    if opt != 'b':
        dilation *= 4
    line = binary_dilation(line, iterations=dilation)
    return line

def remove_nose(image, landmarks):
    ##l: left eye; r: right eye, b: both eye

    (x1, y1), (x2, y2) = landmarks[27], landmarks[30]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(np.array(mask, dtype=np.uint8), (int(x1), int(y1)), (int(x2), int(y2)), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line1 = binary_dilation(line, iterations=dilation)

    (x1, y1), (x2, y2) = landmarks[31], landmarks[35]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(np.array(mask, dtype=np.uint8), (int(x1), int(y1)), (int(x2), int(y2)), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w //4 )
    line2 = binary_dilation(line, iterations=dilation)

    return line1+line2