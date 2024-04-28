import numpy

from .umeyama import umeyama
from numpy.linalg import inv
import cv2

mean_face_x = numpy.array([
0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
0.553364, 0.490127, 0.42689 ])

mean_face_y = numpy.array([
0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
0.784792, 0.824182, 0.831803, 0.824182 ])

landmarks_2D = numpy.stack( [ mean_face_x, mean_face_y ], axis=1 )

def get_align_mat(face, size, should_align_eyes):
    mat_umeyama = umeyama(numpy.array(face.landmarks_as_xy()[17:]), landmarks_2D, True)[0:2]

    if should_align_eyes is False:
        return mat_umeyama

    mat_umeyama = mat_umeyama * size

    # Convert to matrix
    landmarks = numpy.matrix(face.landmarks_as_xy())

    # cv2 expects points to be in the form np.array([ [[x1, y1]], [[x2, y2]], ... ]), we'll expand the dim
    landmarks = numpy.expand_dims(landmarks, axis=1)

    # Align the landmarks using umeyama
    umeyama_landmarks = cv2.transform(landmarks, mat_umeyama, landmarks.shape)

    # Determine a rotation matrix to align eyes horizontally
    mat_align_eyes = align_eyes(umeyama_landmarks, size)

    # Extend the 2x3 transform matrices to 3x3 so we can multiply them
    # and combine them as one
    mat_umeyama = numpy.matrix(mat_umeyama)
    mat_umeyama.resize((3, 3))
    mat_align_eyes = numpy.matrix(mat_align_eyes)
    mat_align_eyes.resize((3, 3))
    mat_umeyama[2] = mat_align_eyes[2] = [0, 0, 1]

    # Combine the umeyama transform with the extra rotation matrix
    transform_mat = mat_align_eyes * mat_umeyama

    # Remove the extra row added, shape needs to be 2x3
    transform_mat = numpy.delete(transform_mat, 2, 0)
    transform_mat = transform_mat / size
    return transform_mat


from .face_blend import get_5_keypoint

def get_align_mat_new(src_lmk, tgt_lmk, size=256, should_align_eyes=False):
    mat_umeyama = umeyama(get_5_keypoint(src_lmk), get_5_keypoint(tgt_lmk), True)[0:2]
    # mat_umeyama = umeyama(numpy.array(src_lmk[17:]), numpy.array(tgt_lmk[17:]), True)[0:2]

    if should_align_eyes is False:
        return mat_umeyama

    mat_umeyama = mat_umeyama * size

    # Convert to matrix
    landmarks = numpy.matrix(face.landmarks_as_xy())

    # cv2 expects points to be in the form np.array([ [[x1, y1]], [[x2, y2]], ... ]), we'll expand the dim
    landmarks = numpy.expand_dims(landmarks, axis=1)

    # Align the landmarks using umeyama
    umeyama_landmarks = cv2.transform(landmarks, mat_umeyama, landmarks.shape)

    # Determine a rotation matrix to align eyes horizontally
    mat_align_eyes = align_eyes(umeyama_landmarks, size)

    # Extend the 2x3 transform matrices to 3x3 so we can multiply them
    # and combine them as one
    mat_umeyama = numpy.matrix(mat_umeyama)
    mat_umeyama.resize((3, 3))
    mat_align_eyes = numpy.matrix(mat_align_eyes)
    mat_align_eyes.resize((3, 3))
    mat_umeyama[2] = mat_align_eyes[2] = [0, 0, 1]

    # Combine the umeyama transform with the extra rotation matrix
    transform_mat = mat_align_eyes * mat_umeyama

    # Remove the extra row added, shape needs to be 2x3
    transform_mat = numpy.delete(transform_mat, 2, 0)
    transform_mat = transform_mat / size
    return transform_mat

# Code borrowed from https://github.com/jrosebr1/imutils/blob/d5cb29d02cf178c399210d5a139a821dfb0ae136/imutils/face_utils/helpers.py
"""
The MIT License (MIT)

Copyright (c) 2015-2016 Adrian Rosebrock, http://www.pyimagesearch.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from collections import OrderedDict
import numpy as np
import cv2

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17)),
    ("chin", (8, 11))
])

# Returns a rotation matrix that when applied to the 68 input facial landmarks
# results in landmarks with eyes aligned horizontally
def align_eyes(landmarks, size):
    desiredLeftEye = (0.35, 0.35) # (y, x) value
    desiredFaceWidth = desiredFaceHeight = size

    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = landmarks[lStart:lEnd]
    rightEyePts = landmarks[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[0,1] - leftEyeCenter[0,1]
    dX = rightEyeCenter[0,0] - leftEyeCenter[0,0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0,0] + rightEyeCenter[0,0]) // 2, (leftEyeCenter[0,1] + rightEyeCenter[0,1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, 1.0)

    return M