# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE
import logging
import os
import pickle

import cv2
import numpy as np
import scipy as sp
import yaml
from skimage.measure import label, regionprops
import random
from PIL import Image
import sys
import albumentations as A
from torch.utils.data import DataLoader
from dataset.utils.bi_online_generation import random_get_hull
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.pair_dataset import pairDataset
import torch

class RandomDownScale(A.core.transforms_interface.ImageOnlyTransform):
    def apply(self, img, ratio_list=None, **params):
        if ratio_list is None:
            ratio_list = [2, 4]
        r = ratio_list[np.random.randint(len(ratio_list))]
        return self.randomdownscale(img, r)

    def randomdownscale(self, img, r):
        keep_ratio = True
        keep_input_shape = True
        H, W, C = img.shape

        img_ds = cv2.resize(img, (int(W / r), int(H / r)), interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_LINEAR)

        return img_ds


'''
from PIL import ImageDraw
# 创建一个可以在图像上绘制的对象
img_pil=Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)

# 在图像上绘制点
for i, point in enumerate(landmark):
    x, y = point
    radius = 1  # 点的半径
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="red")
    draw.text((x+radius+2, y-radius), str(i), fill="black")  # 在点旁边添加标签
img_pil.show()

'''

def alpha_blend(source, target, mask):
    mask_blured = get_blend_mask(mask)
    img_blended = (mask_blured * source + (1 - mask_blured) * target)
    return img_blended, mask_blured


def dynamic_blend(source, target, mask):
    mask_blured = get_blend_mask(mask)
    # worth consideration, 1 in the official paper, 0.25, 0.5, 0.75,1,1,1 in sbi.
    blend_list = [1, 1, 1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask_blured *= blend_ratio
    img_blended = (mask_blured * source + (1 - mask_blured) * target)
    return img_blended, mask_blured


def get_blend_mask(mask):
    H, W = mask.shape
    size_h = np.random.randint(192, 257)
    size_w = np.random.randint(192, 257)
    mask = cv2.resize(mask, (size_w, size_h))
    kernel_1 = random.randrange(5, 26, 2)
    kernel_1 = (kernel_1, kernel_1)
    kernel_2 = random.randrange(5, 26, 2)
    kernel_2 = (kernel_2, kernel_2)

    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured[mask_blured < 1] = 0

    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5, 46))
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured = cv2.resize(mask_blured, (W, H))
    return mask_blured.reshape((mask_blured.shape + (1,)))


def get_alpha_blend_mask(mask):
    kernel_list = [(11, 11), (9, 9), (7, 7), (5, 5), (3, 3)]
    blend_list = [0.25, 0.5, 0.75]
    kernel_idxs = random.choices(range(len(kernel_list)), k=2)
    blend_ratio = blend_list[random.sample(range(len(blend_list)), 1)[0]]
    mask_blured = cv2.GaussianBlur(mask, kernel_list[0], 0)
    # print(mask_blured.max())
    mask_blured[mask_blured < mask_blured.max()] = 0
    mask_blured[mask_blured > 0] = 1
    # mask_blured = mask
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_list[kernel_idxs[1]], 0)
    mask_blured = mask_blured / (mask_blured.max())
    return mask_blured.reshape((mask_blured.shape + (1,)))


class I2GDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        #config['GridShuffle']['p'] = 0
        super().__init__(config, mode)
        real_images_list = [img for img, label in zip(self.image_list, self.label_list) if label == 0]
        self.real_images_list = list(set(real_images_list))  #  de-duplicate since DF,F2F,FS,NT have same real images
        self.source_transforms = self.get_source_transforms()
        self.transforms = self.get_transforms()
        self.init_nearest()

    def init_nearest(self):
        if os.path.exists('training/lib/nearest_face_info.pkl'):
            with open('training/lib/nearest_face_info.pkl', 'rb') as f:
                face_info = pickle.load(f)
        self.face_info = face_info
        # Check if the dictionary has already been created
        if os.path.exists('training/lib/landmark_dict_ffall.pkl'):
            with open('training/lib/landmark_dict_ffall.pkl', 'rb') as f:
                landmark_dict = pickle.load(f)
        self.landmark_dict = landmark_dict

    def reorder_landmark(self, landmark):
        landmark = landmark.copy()  # 创建landmark的副本
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark

    def hflip(self, img, mask=None, landmark=None, bbox=None):
        H, W = img.shape[:2]
        landmark = landmark.copy()
        if bbox is not None:
            bbox = bbox.copy()

        if landmark is not None:
            landmark_new = np.zeros_like(landmark)

            landmark_new[:17] = landmark[:17][::-1]
            landmark_new[17:27] = landmark[17:27][::-1]

            landmark_new[27:31] = landmark[27:31]
            landmark_new[31:36] = landmark[31:36][::-1]

            landmark_new[36:40] = landmark[42:46][::-1]
            landmark_new[40:42] = landmark[46:48][::-1]

            landmark_new[42:46] = landmark[36:40][::-1]
            landmark_new[46:48] = landmark[40:42][::-1]

            landmark_new[48:55] = landmark[48:55][::-1]
            landmark_new[55:60] = landmark[55:60][::-1]

            landmark_new[60:65] = landmark[60:65][::-1]
            landmark_new[65:68] = landmark[65:68][::-1]
            if len(landmark) == 68:
                pass
            elif len(landmark) == 81:
                landmark_new[68:81] = landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:, 0] = W - landmark_new[:, 0]

        else:
            landmark_new = None

        if bbox is not None:
            bbox_new = np.zeros_like(bbox)
            bbox_new[0, 0] = bbox[1, 0]
            bbox_new[1, 0] = bbox[0, 0]
            bbox_new[:, 0] = W - bbox_new[:, 0]
            bbox_new[:, 1] = bbox[:, 1].copy()
            if len(bbox) > 2:
                bbox_new[2, 0] = W - bbox[3, 0]
                bbox_new[2, 1] = bbox[3, 1]
                bbox_new[3, 0] = W - bbox[2, 0]
                bbox_new[3, 1] = bbox[2, 1]
                bbox_new[4, 0] = W - bbox[4, 0]
                bbox_new[4, 1] = bbox[4, 1]
                bbox_new[5, 0] = W - bbox[6, 0]
                bbox_new[5, 1] = bbox[6, 1]
                bbox_new[6, 0] = W - bbox[5, 0]
                bbox_new[6, 1] = bbox[5, 1]
        else:
            bbox_new = None

        if mask is not None:
            mask = mask[:, ::-1]
        else:
            mask = None
        img = img[:, ::-1].copy()
        return img, mask, landmark_new, bbox_new



    def get_source_transforms(self):
        return A.Compose([
            A.Compose([
                A.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                A.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                     val_shift_limit=(-0.3, 0.3), p=1),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
            ], p=1),

            A.OneOf([
                RandomDownScale(p=1),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=1),

        ], p=1.)

    def get_fg_bg(self, one_lmk_path):
        """
        Get foreground and background paths
        """
        bg_lmk_path = one_lmk_path
        # Randomly pick one from the nearest neighbors for the foreground
        if bg_lmk_path in self.face_info:
            fg_lmk_path = random.choice(self.face_info[bg_lmk_path])
        else:
            fg_lmk_path = bg_lmk_path
        return fg_lmk_path, bg_lmk_path

    def get_transforms(self):
        return A.Compose([

            A.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            A.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                 val_shift_limit=(-0.3, 0.3), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),

        ],
            additional_targets={f'image1': 'image'},
            p=1.)

    def randaffine(self, img, mask):
        f = A.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1)

        g = A.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']

        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask

    def __len__(self):
        return len(self.real_images_list)


    def colorTransfer(self, src, dst, mask):
        transferredDst = np.copy(dst)
        maskIndices = np.where(mask != 0)
        maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.float32)
        maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.float32)

        # Compute means and standard deviations
        meanSrc = np.mean(maskedSrc, axis=0)
        stdSrc = np.std(maskedSrc, axis=0)
        meanDst = np.mean(maskedDst, axis=0)
        stdDst = np.std(maskedDst, axis=0)

        # Perform color transfer
        maskedDst = (maskedDst - meanDst) * (stdSrc / stdDst) + meanSrc
        maskedDst = np.clip(maskedDst, 0, 255)

        # Copy the entire background into transferredDst
        transferredDst = np.copy(dst)
        # Now apply color transfer only to the masked region
        transferredDst[maskIndices[0], maskIndices[1]] = maskedDst.astype(np.uint8)

        return transferredDst



    def two_blending(self, img_bg, img_fg, landmark):
        H, W = len(img_bg), len(img_bg[0])
        if np.random.rand() < 0.25:
            landmark = landmark[:68]
        logging.disable(logging.FATAL)
        mask = random_get_hull(landmark, img_bg)
        logging.disable(logging.NOTSET)
        source = img_fg.copy()
        target = img_bg.copy()
        # if np.random.rand() < 0.5:
        #     source = self.source_transforms(image=source.astype(np.uint8))['image']
        # else:
        #     target = self.source_transforms(image=target.astype(np.uint8))['image']
        source_v2, mask_v2 = self.randaffine(source, mask)
        source_v3=self.colorTransfer(target,source_v2,mask_v2)
        img_blended, mask = dynamic_blend(source_v3, target, mask_v2)
        img_blended = img_blended.astype(np.uint8)
        img = img_bg.astype(np.uint8)

        return img, img_blended, mask.squeeze(2)


    def __getitem__(self, index):
        image_path_bg = self.real_images_list[index]
        label = 0

        # Get the mask and landmark paths
        landmark_path_bg = image_path_bg.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark
        landmark_path_fg, landmark_path_bg = self.get_fg_bg(landmark_path_bg)
        image_path_fg = landmark_path_fg.replace('landmarks','frames').replace('.npy','.png')
        try:
            image_bg = self.load_rgb(image_path_bg)
            image_fg = self.load_rgb(image_path_fg)
        except Exception as e:
            # Skip this image and return the first one
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image_bg = np.array(image_bg)  # Convert to numpy array for data augmentation
        image_fg = np.array(image_fg)  # Convert to numpy array for data augmentation

        landmarks_bg = self.load_landmark(landmark_path_bg)
        landmarks_fg = self.load_landmark(landmark_path_fg)


        landmarks_bg = np.clip(landmarks_bg, 0, self.config['resolution'] - 1)
        landmarks_bg = self.reorder_landmark(landmarks_bg)

        img_r, img_f, mask_f = self.two_blending(image_bg.copy(), image_fg.copy(),landmarks_bg.copy())
        transformed = self.transforms(image=img_f.astype('uint8'), image1=img_r.astype('uint8'))
        img_f = transformed['image']
        img_r = transformed['image1']
        # img_f = img_f.transpose((2, 0, 1))
        # img_r = img_r.transpose((2, 0, 1))
        img_f = self.normalize(self.to_tensor(img_f))
        img_r = self.normalize(self.to_tensor(img_r))
        mask_f = self.to_tensor(mask_f)
        mask_r=torch.zeros_like(mask_f) # zeros or ones
        return img_f, img_r, mask_f,mask_r

    @staticmethod
    def collate_fn(batch):
        img_f, img_r, mask_f,mask_r = zip(*batch)
        data = {}
        fake_mask = torch.stack(mask_f,dim=0)
        real_mask = torch.stack(mask_r, dim=0)
        fake_images = torch.stack(img_f, dim=0)
        real_images = torch.stack(img_r, dim=0)
        data['image'] = torch.cat([real_images, fake_images], dim=0)
        data['label'] = torch.tensor([0] * len(img_r) + [1] * len(img_f))
        data['landmark'] = None
        data['mask'] = torch.cat([real_mask, fake_mask], dim=0)
        return data


if __name__ == '__main__':
    detector_path = r"./training/config/detector/xception.yaml"
    # weights_path = "./ckpts/xception/CDFv2/tb_v1/ov.pth"
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config2['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config.update(config2)
    dataset = I2GDataset(config=config)
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=dataset.collate_fn)

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {batch}")
        continue

