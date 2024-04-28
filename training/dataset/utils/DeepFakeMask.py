#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Created by: algohunt
# Microsoft Research & Peking University 
# lilingzhi@pku.edu.cn
# Copyright (c) 2019

#!/usr/bin/env python3
""" Masks functions for faceswap.py """

import inspect
import logging
import sys

import cv2
import numpy as np
import random
from math import ceil, floor
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def landmarks_to_bbox(landmarks: np.ndarray) -> np.ndarray:
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    assert landmarks.shape[1] == 2
    x0, y0 = np.min(landmarks, axis=0) # x和y轴上分别的最小值, [264,97]
    x1, y1 = np.max(landmarks, axis=0) # x和y轴上分别的最小值, [370,236]
    bbox = np.array([x0, y0, x1, y1])
    return bbox 

def mask_from_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """8 (or omitted) - 8-connected line.
          4 - 4-connected line.
    LINE_AA - antialiased line."""
    h, w = image.shape[:2]
    points = points.astype(int)
    assert points.shape[1] == 2, f"points.shape: {points.shape}"
    out = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(points.astype(int))
    cv2.fillConvexPoly(out, hull, 255, lineType=4)  # cv2.LINE_AA
    return out

def get_available_masks():
    """ Return a list of the available masks for cli """
    masks = sorted([name for name, obj in inspect.getmembers(sys.modules[__name__])
                    if inspect.isclass(obj) and name != "Mask"])
    masks.append("none")
    # logger.debug(masks)
    return masks

def landmarks_68_symmetries():
    # 68 landmarks symmetry
    #
    sym_ids = [9, 58, 67, 63, 52, 34, 31, 30, 29, 28]
    sym = {
        1: 17,
        2: 16,
        3: 15,
        4: 14,
        5: 13,
        6: 12,
        7: 11,
        8: 10,
        #
        51: 53,
        50: 54,
        49: 55,
        60: 56,
        59: 57,
        #
        62: 64,
        61: 65,
        68: 66,
        #
        33: 35,
        32: 36,
        #
        37: 46,
        38: 45,
        39: 44,
        40: 43,
        41: 48,
        42: 47,
        #
        18: 27,
        19: 26,
        20: 25,
        21: 24,
        22: 23,
        #
        # id
        9: 9,
        58: 58,
        67: 67,
        63: 63,
        52: 52,
        34: 34,
        31: 31,
        30: 30,
        29: 29,
        28: 28,
    }
    return sym, sym_ids



def get_default_mask():
    """ Set the default mask for cli """
    masks = get_available_masks()
    default = "dfl_full"
    default = default if default in masks else masks[0]
    # logger.debug(default)
    return default


class Mask():
    """ Parent class for masks
        the output mask will be <mask_type>.mask
        channels: 1, 3 or 4:
                    1 - Returns a single channel mask
                    3 - Returns a 3 channel mask
                    4 - Returns the original image with the mask in the alpha channel """

    def __init__(self, landmarks, face, channels=4, idx = 0):
        # logger.info("Initializing %s: (face_shape: %s, channels: %s, landmarks: %s)",
        #              self.__class__.__name__, face.shape, channels, landmarks)
        self.landmarks = landmarks
        self.face = face
        self.channels = channels
        self.cols = 4 # grid mask
        self.rows = 4 # grid mask
        self.idx = idx # grid mask

        mask = self.build_mask()
        self.mask = self.merge_mask(mask)
        # logger.info("Initialized %s", self.__class__.__name__)

    def build_mask(self):
        """ Override to build the mask """
        raise NotImplementedError

    def merge_mask(self, mask):
        """ Return the mask in requested shape """
        # logger.info("mask_shape: %s", mask.shape)
        assert self.channels in (1, 3, 4), "Channels should be 1, 3 or 4"
        assert mask.shape[2] == 1 and mask.ndim == 3, "Input mask be 3 dimensions with 1 channel"

        if self.channels == 3:
            retval = np.tile(mask, 3)
        elif self.channels == 4:
            retval = np.concatenate((self.face, mask), -1)
        else:
            retval = mask

        # logger.info("Final mask shape: %s", retval.shape)
        return retval


class dfl_full(Mask):  # pylint: disable=invalid-name
    """ DFL facial mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        nose_ridge = (self.landmarks[27:31], self.landmarks[33:34])
        jaw = (self.landmarks[0:17],
               self.landmarks[48:68],
               self.landmarks[0:1],
               self.landmarks[8:9],
               self.landmarks[16:17])
        eyes = (self.landmarks[17:27],
                self.landmarks[0:1],
                self.landmarks[27:28],
                self.landmarks[16:17],
                self.landmarks[33:34])
        parts = [jaw, nose_ridge, eyes]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


class components(Mask):  # pylint: disable=invalid-name
    """ Component model mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        r_jaw = (self.landmarks[0:9], self.landmarks[17:18])
        l_jaw = (self.landmarks[8:17], self.landmarks[26:27])
        r_cheek = (self.landmarks[17:20], self.landmarks[8:9])
        l_cheek = (self.landmarks[24:27], self.landmarks[8:9])
        nose_ridge = (self.landmarks[19:25], self.landmarks[8:9],)
        r_eye = (self.landmarks[17:22],
                 self.landmarks[27:28],
                 self.landmarks[31:36],
                 self.landmarks[8:9])
        l_eye = (self.landmarks[22:27],
                 self.landmarks[27:28],
                 self.landmarks[31:36],
                 self.landmarks[8:9])
        nose = (self.landmarks[27:31], self.landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

        # ---change 0531 random select parts ---
        # r_face = (self.landmarks[0:9], self.landmarks[17:18],self.landmarks[17:20], self.landmarks[8:9])
        # l_face = (self.landmarks[8:17], self.landmarks[26:27],self.landmarks[24:27], self.landmarks[8:9])
        # nose_final = (self.landmarks[19:25], self.landmarks[8:9],self.landmarks[27:31], self.landmarks[31:36])
        # parts = [r_face,l_face,nose_final,r_eye,l_eye]
        # num_to_select = random.randint(1, len(parts))
        # parts = random.sample(parts, num_to_select)
        # print(len(parts), parts[0])
        # ---change 0531 random select parts ---

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


class extended(Mask):  # pylint: disable=invalid-name
    """ Extended mask
        Based on components mask. Attempts to extend the eyebrow points up the forehead
    """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        landmarks = self.landmarks.copy()
        # mid points between the side of face and eye point
        ml_pnt = (landmarks[36] + landmarks[0]) // 2
        mr_pnt = (landmarks[16] + landmarks[45]) // 2

        # mid points between the mid points and eye
        ql_pnt = (landmarks[36] + ml_pnt) // 2
        qr_pnt = (landmarks[45] + mr_pnt) // 2

        # Top of the eye arrays
        bot_l = np.array((ql_pnt, landmarks[36], landmarks[37], landmarks[38], landmarks[39]))
        bot_r = np.array((landmarks[42], landmarks[43], landmarks[44], landmarks[45], qr_pnt))

        # Eyebrow arrays
        top_l = landmarks[17:22]
        top_r = landmarks[22:27]

        # Adjust eyebrow arrays
        landmarks[17:22] = top_l + ((top_l - bot_l) // 2)
        landmarks[22:27] = top_r + ((top_r - bot_r) // 2)

        r_jaw = (landmarks[0:9], landmarks[17:18])
        l_jaw = (landmarks[8:17], landmarks[26:27])
        r_cheek = (landmarks[17:20], landmarks[8:9])
        l_cheek = (landmarks[24:27], landmarks[8:9])
        nose_ridge = (landmarks[19:25], landmarks[8:9],)
        r_eye = (landmarks[17:22], landmarks[27:28], landmarks[31:36], landmarks[8:9])
        l_eye = (landmarks[22:27], landmarks[27:28], landmarks[31:36], landmarks[8:9])
        nose = (landmarks[27:31], landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


class facehull(Mask):  # pylint: disable=invalid-name
    """ Basic face hull mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)
        hull = cv2.convexHull(  # pylint: disable=no-member
            np.array(self.landmarks).reshape((-1, 2)))
        cv2.fillConvexPoly(mask, hull, 255.0, lineType=cv2.LINE_AA)  # pylint: disable=no-member
        return mask
        # mask = np.zeros(img.shape[0:2] + (1, ), dtype=np.float32)
        # hull = cv2.convexHull(np.array(landmark).reshape((-1, 2)))

class facehull2(Mask):  # pylint: disable=invalid-name
    """ Basic face hull mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.uint8)
        hull = cv2.convexHull(  # pylint: disable=no-member
            np.array(self.landmarks).reshape((-1, 2)))
        cv2.fillConvexPoly(mask, hull, 1.0, lineType=cv2.LINE_AA)
        return mask



class gridMasking(Mask):

    def build_mask(self):
        h, w = self.face.shape[:2]
        landmarks = self.landmarks[:68]
        # if idx is None:
        #    idx = np.random.randint(0, self.total)
        r, c = divmod(self.idx, self.cols) # 获得除数和余数，即这个idx对应第r行第c列

        # pixel related
        xmin, ymin, xmax, ymax = landmarks_to_bbox(landmarks)
        dx = ceil((xmax - xmin) / self.cols)
        dy = ceil((ymax - ymin) / self.rows)

        mask = np.zeros((h, w), dtype=np.uint8)

        # fill the cell mask
        x0, y0 = floor(xmin + dx * c), floor(ymin + dy * r)
        x1, y1 = floor(x0 + dx), floor(y0 + dy)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

        # merge the cell mask with the convex hull
        ch = mask_from_points(self.face, landmarks)
        # ch = cv2.cvtColor(ch, cv2.COLOR_BGR2GRAY)
        # mask = (mask & ch) / 255.0
        mask = cv2.bitwise_and(mask, mask, mask=ch)
        mask = mask.reshape([mask.shape[0],mask.shape[1], 1])
        # cv2.bitwise_or(img, d_3c_i)

        return mask

class MeshgridMasking(Mask):
    areas = [
        [1, 2, 3, 4, 5, 6, 7, 49, 32, 40, 41, 42, 37, 18],
        [37, 38, 39, 40, 41, 42],  # left eye
        [18, 19, 20, 21, 22, 28, 40, 39, 38, 37],
        [28, 29, 30, 31, 32, 40],
    ]
    areas_asym = [
        [20, 21, 22, 28, 23, 24, 25],  # old [22, 23, 28],
        [31, 32, 33, 34, 35, 36],
        [32, 33, 34, 35, 36, 55, 54, 53, 52, 51, 50, 49],
        [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
        [7, 8, 9, 10, 11, 55, 56, 57, 58, 59, 60, 49],
    ]

    def init(self, **kwargs):
        # super().__init__(**kwargs)

        sym, _ = landmarks_68_symmetries()
        # construct list of points paths
        paths = []
        paths += self.areas_asym  # asymmetrical areas
        paths += self.areas  # left
        paths += [[sym[ld68_id] for ld68_id in area] for area in self.areas]  # right
        assert len(paths) == self.total
        self.paths = paths

    @property
    def total(self) -> int:
        total = len(self.areas_asym) + len(self.areas) * 2
        return total

    def transform_landmarks(self, landmarks):
        """Transform landmarks to extend the eyebrow points up the forehead"""
        new_landmarks = landmarks.copy()
        # mid points between the side of face and eye point
        ml_pnt = (new_landmarks[36] + new_landmarks[0]) // 2
        mr_pnt = (new_landmarks[16] + new_landmarks[45]) // 2

        # mid points between the mid points and eye
        ql_pnt = (new_landmarks[36] + ml_pnt) // 2
        qr_pnt = (new_landmarks[45] + mr_pnt) // 2

        # Top of the eye arrays
        bot_l = np.array(
            (
                ql_pnt,
                new_landmarks[36],
                new_landmarks[37],
                new_landmarks[38],
                new_landmarks[39],
            )
        )
        bot_r = np.array(
            (
                new_landmarks[42],
                new_landmarks[43],
                new_landmarks[44],
                new_landmarks[45],
                qr_pnt,
            )
        )

        # Eyebrow arrays
        top_l = new_landmarks[17:22]
        top_r = new_landmarks[22:27]

        # Adjust eyebrow arrays
        new_landmarks[17:22] = top_l + ((top_l - bot_l) // 2)
        new_landmarks[22:27] = top_r + ((top_r - bot_r) // 2)

        return new_landmarks

    def build_mask(self) -> np.ndarray:
        self.init()
        h, w = self.face.shape[:2]

        path = self.paths[self.idx]
        new_landmarks = self.transform_landmarks(self.landmarks)
        points = [new_landmarks[ld68_id - 1] for ld68_id in path]
        points = np.array(points, dtype=np.int32)

        # cv2.fillConvexPoly(out, points, 255, lineType=4)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        mask = mask.reshape([mask.shape[0],mask.shape[1], 1])
        return mask