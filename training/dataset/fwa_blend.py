import os
import sys
import json
import pickle
import time
import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from skimage.util import random_noise
from skimage.draw import polygon
from scipy import linalg
import heapq as hq

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T
import torchvision

from dataset.utils.face_blend import *
from dataset.utils.face_align import get_align_mat_new
from dataset.utils.color_transfer import color_transfer
from dataset.utils.faceswap_utils import blendImages as alpha_blend_fea
from dataset.utils.faceswap_utils import AlphaBlend as alpha_blend
from dataset.utils.face_aug import aug_one_im, change_res
from dataset.utils.image_ae import get_pretraiend_ae
from dataset.utils.warp import warp_mask
from dataset.utils import faceswap
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import AffineTransform, warp

from dataset.ff_blend import FFBlendDataset


# Define face detector and predictor models
face_detector = dlib.get_frontal_face_detector()
predictor_path = '/home/yuanxinhang/SelfBlendedImages/src/preprocess/shape_predictor_81_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)


mean_face_x = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689])

mean_face_y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
    0.784792, 0.824182, 0.831803, 0.824182])

landmarks_2D = np.stack([mean_face_x, mean_face_y], axis=1)


def umeyama( src, dst, estimate_scale ):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


from skimage.transform import AffineTransform, warp

def get_warped_face(face, landmarks, tform):
    """
    Apply the given affine transformation to the face and landmarks.

    Args:
        face (np.ndarray): The face image to be transformed.
        landmarks (np.ndarray): The facial landmarks to be transformed.
        tform (AffineTransform): The transformation to apply.

    Returns:
        warped_face (np.ndarray): The transformed face image.
        warped_landmarks (np.ndarray): The transformed facial landmarks.
    """
    # Apply the transformation to the face
    warped_face = warp(face, tform.inverse, output_shape=face.shape)
    warped_face = (warped_face * 255).astype(np.uint8)

    # Apply the transformation to the landmarks
    warped_landmarks = tform.inverse(landmarks)

    return warped_face, warped_landmarks


def warp_face_within_landmarks(face, landmarks, tform):
    """
    Apply the given affine transformation to the face and landmarks, 
    and retain only the area within the landmarks.

    Args:
        face (np.ndarray): The face image to be transformed.
        landmarks (np.ndarray): The facial landmarks to be transformed.
        tform (AffineTransform): The transformation to apply.

    Returns:
        warped_face (np.ndarray): The transformed face image.
        warped_landmarks (np.ndarray): The transformed facial landmarks.
    """
    # Apply the transformation to the face
    warped_face = warp(face, tform.inverse, output_shape=face.shape)
    warped_face = (warped_face * 255).astype(np.uint8)

    # Apply the transformation to the landmarks
    warped_landmarks = np.linalg.inv(landmarks)

    # Generate a mask based on the landmarks
    rr, cc = polygon(warped_landmarks[:, 1], warped_landmarks[:, 0])
    mask = np.zeros_like(warped_face, dtype=np.uint8)
    mask[rr, cc] = 1

    # Apply the mask to the face
    warped_face *= mask

    return warped_face, warped_landmarks


def get_2d_aligned_face(image, mat, size, padding=[0, 0]):
    mat = mat * size
    mat[0, 2] += padding[0]
    mat[1, 2] += padding[1]
    return cv2.warpAffine(image, mat, (size + 2 * padding[0], size + 2 * padding[1]))


def get_2d_aligned_landmarks(face_cache, aligned_face_size=256, padding=(0, 0)):
    mat, points = face_cache
    # Mapping landmarks to aligned face
    pred_ = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
    pred_ = np.transpose(pred_)
    mat = mat * aligned_face_size
    mat[0, 2] += padding[0]
    mat[1, 2] += padding[1]
    aligned_shape = np.dot(mat, pred_)
    aligned_shape = np.transpose(aligned_shape[:2, :])
    return aligned_shape


def get_aligned_face_and_landmarks(im, face_cache, aligned_face_size = 256, padding=(0, 0)):
    """
    get all aligned faces and landmarks of all images
    :param imgs: origin images
    :param fa: face_alignment package
    :return:
    """
    aligned_cur_shapes = []
    aligned_cur_im = []
    for mat, points in face_cache:
        # Get transform matrix
        aligned_face = get_2d_aligned_face(im, mat, aligned_face_size, padding)
        aligned_shape = get_2d_aligned_landmarks([mat, points], aligned_face_size, padding)
        aligned_cur_shapes.append(aligned_shape)
        aligned_cur_im.append(aligned_face)
    return aligned_cur_im, aligned_cur_shapes


def face_warp(im, face, trans_matrix, size, padding):
    new_face = np.clip(face, 0, 255).astype(im.dtype)
    image_size = im.shape[1], im.shape[0]

    tmp_matrix = trans_matrix * size
    delta_matrix = np.array([[0., 0., padding[0]*1.0], [0., 0., padding[1]*1.0]])
    tmp_matrix = tmp_matrix + delta_matrix

    # Warp the new face onto a blank canvas
    warped_face = np.zeros_like(im)
    cv2.warpAffine(new_face, tmp_matrix, image_size, warped_face, cv2.WARP_INVERSE_MAP,
                   cv2.BORDER_TRANSPARENT)
    
    # Create a mask of the warped face
    mask = (warped_face > 0).astype(np.uint8)

    # Blend the warped face with the original image
    new_image = im * (1 - mask) + warped_face * mask

    return new_image, mask


def get_face_loc(im, face_detector, scale=0):
    """ get face locations, color order of images is rgb """
    faces = face_detector(np.uint8(im), scale)
    face_list = []
    if faces is not None or len(faces) > 0:
        for i, d in enumerate(faces):
            try:
                face_list.append([d.left(), d.top(), d.right(), d.bottom()])
            except:
                face_list.append([d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()])
    return face_list



def align(im, face_detector, lmark_predictor, scale=0):
    # This version we handle all faces in view
    # channel order rgb
    im = np.uint8(im)
    faces = face_detector(im, scale)
    face_list = []
    if faces is not None or len(faces) > 0:
        for pred in faces:
            try:
                points = shape_to_np(lmark_predictor(im, pred))
            except:
                points = shape_to_np(lmark_predictor(im, pred.rect))
            trans_matrix = umeyama(points[17:], landmarks_2D, True)[0:2]
            face_list.append([trans_matrix, points])
    return face_list


class FWABlendDataset(FFBlendDataset):
    def __init__(self, config=None):
        # Check if the dictionary has already been created
        if os.path.exists('nearest_face_info_new.pkl'):
            with open('nearest_face_info_new.pkl', 'rb') as f:
                face_info = pickle.load(f)
        self.face_info = face_info
        # Check if the dictionary has already been created
        if os.path.exists('landmark_dict_ffall.pkl'):
            with open('landmark_dict_ffall.pkl', 'rb') as f:
                landmark_dict = pickle.load(f)
        self.landmark_dict = landmark_dict
        self.imid_list = self.get_training_imglist()
        self.transforms = T.Compose([
            # T.RandomHorizontalFlip(),
            # T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # T.ColorJitter(brightness=0.1, contrast=0.1,
            #              saturation=0.1, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])
        self.data_dict = {
            'img_list': self.imid_list
        }

    
    def get_training_imglist(self):
        """
        Get the list of training images.
        """
        random.seed(1024)  # Fix the random seed for reproducibility
        imid_list = list(self.landmark_dict.keys())
        # imid_list = [imid.replace('landmarks', 'frames').replace('npy', 'png') for imid in imid_list]
        random.shuffle(imid_list)
        return imid_list


    def preprocess_images(self, imid_fg, imid_bg):
        """
        Load foreground and background images and face shapes, and apply transformations.
        """
        im = cv2.imread(imid_fg.replace('landmarks', 'frames').replace('npy', 'png'))

        # Get the alignment of the head
        face_cache = align(im, face_detector, face_predictor)

        # Get the aligned face and landmarks
        aligned_im_head, aligned_shape = get_aligned_face_and_landmarks(im, face_cache)
        # If no faces were detected in the image, return None (or any suitable value)
        if len(aligned_im_head) == 0 or len(aligned_shape) == 0:
            return None, None
        aligned_im_head = aligned_im_head[0]
        aligned_shape = aligned_shape[0]

        # Apply transformations to the face
        scale_factor = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        scaled_face = cv2.resize(aligned_im_head, (0, 0), fx=scale_factor, fy=scale_factor)

        # Apply Gaussian blur to the scaled face
        blurred_face = cv2.GaussianBlur(scaled_face, (5, 5), 0)

        # Resize the processed image back to the original size
        resized_face = cv2.resize(blurred_face, (aligned_im_head.shape[1], aligned_im_head.shape[0]))

        # Generate a random facial mask
        mask = get_mask(aligned_shape.astype(np.float32), resized_face, std=20, deform=True)

        # Apply the mask to the resized face
        masked_face = cv2.bitwise_and(resized_face, resized_face, mask=mask)

        # do aug before warp
        im = np.array(self.blended_aug(im))

        # Warp the face back to the original image
        im, masked_face = face_warp(im, masked_face, face_cache[0][0], 256, [0, 0])
        shape = get_2d_aligned_landmarks(face_cache[0], 256, [0, 0])
        return im, masked_face


    def process_images(self, imid_fg, imid_bg, index):
        """
        Process an image following the data generation pipeline.
        """
        im, mask = self.preprocess_images(imid_fg, imid_bg)

        # Prepare images and titles for the combined image
        imid_fg = cv2.imread(imid_fg.replace('landmarks', 'frames').replace('npy', 'png'))
        imid_fg = np.array(self.data_aug(imid_fg))

        if im is None or mask is None:
            return imid_fg, None

        images = [
            imid_fg, 
            np.where(mask.astype(np.uint8)>0, 255, 0), 
            im,
        ]
        titles = ["Image", "Mask", "Blended Image"]

        # Save the combined image
        os.makedirs('fwa_examples_2', exist_ok=True)
        self.save_combined_image(images, titles, index, f'fwa_examples_2/combined_image_{index}.png')
        return imid_fg, im


    def post_proc(self, img):
        '''
        if self.mode == 'train':
            #if np.random.rand() < 0.5:
            #    img = random_add_noise(img)
                #add_gaussian_noise(img)
            if np.random.rand() < 0.5:
                #img, _ = change_res(img)
                img = gaussian_blur(img)
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_aug = self.blended_aug(img)
        im_aug = Image.fromarray(np.uint8(img))
        im_aug = self.transforms(im_aug)
        return im_aug
    

    @staticmethod
    def save_combined_image(images, titles, index, save_path):
        """
        Save the combined image with titles for each single image.

        Args:
            images (List[np.ndarray]): List of images to be combined.
            titles (List[str]): List of titles for each image.
            index (int): Index of the image.
            save_path (str): Path to save the combined image.
        """
        # Determine the maximum height and width among the images
        max_height = max(image.shape[0] for image in images)
        max_width = max(image.shape[1] for image in images)

        # Create the canvas
        canvas = np.zeros((max_height * len(images), max_width, 3), dtype=np.uint8)

        # Place the images and titles on the canvas
        current_height = 0
        for image, title in zip(images, titles):
            height, width = image.shape[:2]
            
            # Check if image has a third dimension (color channels)
            if image.ndim == 2:
                # If not, add a third dimension
                image = np.tile(image[..., None], (1, 1, 3))

            canvas[current_height : current_height + height, :width] = image
            cv2.putText(
                canvas, title, (10, current_height + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            current_height += height

        # Save the combined image
        cv2.imwrite(save_path, canvas)
    

    def __getitem__(self, index):
        """
        Get an item from the dataset by index.
        """
        one_lmk_path = self.imid_list[index]
        label = 1 if one_lmk_path.split('/')[6]=='manipulated_sequences' else 0
        blend_label = 1
        imid, manipulate_img = self.process_images(one_lmk_path, one_lmk_path, index)

        if manipulate_img is None:
            manipulate_img = deepcopy(imid)
            blend_label = label
        manipulate_img = self.post_proc(manipulate_img)
        imid = self.post_proc(imid)

        # blend data
        fake_data_tuple = (manipulate_img, blend_label)
        # original data
        real_data_tuple = (imid, label)

        return fake_data_tuple, real_data_tuple


    @staticmethod
    def collate_fn(batch):
        """
        Collates batches of data and shuffles the images.
        """
        # Unzip the batch
        fake_data, real_data = zip(*batch)

        # Unzip the fake and real data
        fake_images, fake_labels = zip(*fake_data)
        real_images, real_labels = zip(*real_data)

        # Combine fake and real data
        images = torch.stack(fake_images + real_images)
        labels = torch.tensor(fake_labels + real_labels)

        # Combine images, boundaries, and labels into tuples
        combined_data = list(zip(images, labels))

        # Shuffle the combined data
        random.shuffle(combined_data)

        # Unzip the shuffled data
        images, labels = zip(*combined_data)

        # Create the data dictionary
        data_dict = {
            'image': torch.stack(images),
            'label': torch.tensor(labels),
            'mask': None,
            'landmark': None  # Add your landmark data if available
        }

        return data_dict


    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.imid_list)


if __name__ == "__main__":
    dataset = FFBlendDataset()
    print('dataset lenth: ', len(dataset))

    def tensor2bgr(im):
        img = im.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = (img + 1)/2 * 255
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def tensor2gray(im):
        img = im.squeeze().cpu().numpy()
        img = img * 255
        return img

    for i, data_dict in enumerate(dataset):
        if i > 20:
            break
        if label == 1:
            if not use_mouth:
                img, boudary = im
                cv2.imwrite('{}_whole.png'.format(i), tensor2bgr(img))
                cv2.imwrite('{}_boudnary.png'.format(i), tensor2gray(boudary))
            else:
                img, mouth, boudary = im
                cv2.imwrite('{}_whole.png'.format(i), tensor2bgr(img))
                cv2.imwrite('{}_mouth.png'.format(i), tensor2bgr(mouth))
                cv2.imwrite('{}_boudnary.png'.format(i), tensor2gray(boudary))
