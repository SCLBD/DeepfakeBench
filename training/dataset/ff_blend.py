'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is designed for Face X-ray.
'''

import os
import sys
import json
import pickle
import time

import lmdb
import numpy as np
import albumentations as A
import cv2
import random
from PIL import Image
from skimage.util import random_noise
from scipy import linalg
import heapq as hq
import lmdb
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


class RandomDownScale(A.core.transforms_interface.ImageOnlyTransform):
	def apply(self,img,**params):
		return self.randomdownscale(img)

	def randomdownscale(self,img):
		keep_ratio=True
		keep_input_shape=True
		H,W,C=img.shape
		ratio_list=[2,4]
		r=ratio_list[np.random.randint(len(ratio_list))]
		img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
		if keep_input_shape:
			img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)
		return img_ds


class FFBlendDataset(data.Dataset):
    def __init__(self, config=None):

        self.lmdb = config.get('lmdb', False)
        if self.lmdb:
            lmdb_path = os.path.join(config['lmdb_dir'], f"FaceForensics++_lmdb")
            self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)

        # Check if the dictionary has already been created
        if os.path.exists('training/lib/nearest_face_info.pkl'):
            with open('training/lib/nearest_face_info.pkl', 'rb') as f:
                face_info = pickle.load(f)
        else:
            raise ValueError(f"Need to run the dataset/generate_xray_nearest.py before training the face xray.")
        self.face_info = face_info
        # Check if the dictionary has already been created
        if os.path.exists('training/lib/landmark_dict_ffall.pkl'):
            with open('training/lib/landmark_dict_ffall.pkl', 'rb') as f:
                landmark_dict = pickle.load(f)
        self.landmark_dict = landmark_dict
        self.imid_list = self.get_training_imglist()
        self.transforms = T.Compose([
            # T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # T.ColorJitter(hue=.05, saturation=.05),
            # T.RandomHorizontalFlip(),
            # T.RandomRotation(20, resample=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])
        self.data_dict = {
            'imid_list': self.imid_list
        }
        self.config=config
    # def data_aug(self, im):
    #     """
    #     Apply data augmentation on the input image.
    #     """
    #     transform = T.Compose([
    #         T.ToPILImage(),
    #         T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    #         T.ColorJitter(hue=.05, saturation=.05),
    #     ])
    #     # Apply transformations
    #     im_aug = transform(im)
    #     return im_aug

    def blended_aug(self, im):
        transform = A.Compose([
            A.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            A.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            A.ImageCompression(quality_lower=40, quality_upper=100,p=0.5)
        ])
        # Apply transformations
        im_aug = transform(image=im)
        return im_aug['image']


    def data_aug(self, im):
        """
        Apply data augmentation on the input image using albumentations.
        """
        transform = A.Compose([
            A.Compose([
                A.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
                A.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
            ],p=1),
            A.OneOf([
                RandomDownScale(p=1),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ],p=1),
        ], p=1.)
        # Apply transformations
        im_aug = transform(image=im)
        return im_aug['image']

    
    def get_training_imglist(self):
        """
        Get the list of training images.
        """
        random.seed(1024)  # Fix the random seed for reproducibility
        imid_list = list(self.landmark_dict.keys())
        # imid_list = [imid.replace('landmarks', 'frames').replace('npy', 'png') for imid in imid_list]
        random.shuffle(imid_list)
        return imid_list

    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution'] # if self.mode == "train" else self.config['resolution']
        if not self.lmdb:
            if not file_path[0] == '.':
                file_path =  f'./{self.config["rgb_dir"]}\\'+file_path
            assert os.path.exists(file_path), f"{file_path} does not exist"
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError('Loaded image is None: {}'.format(file_path))
        elif self.lmdb:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')

                image_bin = txn.get(file_path.encode())
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return np.array(img, dtype=np.uint8)


    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            if not file_path[0] == '.':
                file_path =  f'./{self.config["rgb_dir"]}\\'+file_path
            return np.zeros((size, size, 1))
        if not self.lmdb:
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')
                image_bin = txn.get(file_path.encode())
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
                mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if not self.lmdb:
            if not file_path[0] == '.':
                file_path =  f'./{self.config["rgb_dir"]}\\'+file_path
            if os.path.exists(file_path):
                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')
                binary = txn.get(file_path.encode())
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
        return np.float32(landmark)

    def preprocess_images(self, imid_fg, imid_bg):
        """
        Load foreground and background images and face shapes.
        """
        fg_im = self.load_rgb(imid_fg.replace('landmarks', 'frames').replace('npy', 'png'))
        fg_im = np.array(self.data_aug(fg_im))
        fg_shape = self.landmark_dict[imid_fg]
        fg_shape = np.array(fg_shape, dtype=np.int32)

        bg_im = self.load_rgb(imid_bg.replace('landmarks', 'frames').replace('npy', 'png'))
        bg_im = np.array(self.data_aug(bg_im))
        bg_shape = self.landmark_dict[imid_bg]
        bg_shape = np.array(bg_shape, dtype=np.int32)

        if fg_im is None:
            return bg_im, bg_shape, bg_im, bg_shape
        elif bg_im is None:
            return fg_im, fg_shape, fg_im, fg_shape
        
        return fg_im, fg_shape, bg_im, bg_shape


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


    def generate_masks(self, fg_im, fg_shape, bg_im, bg_shape):
        """
        Generate masks for foreground and background images.
        """
        fg_mask = get_mask(fg_shape, fg_im, deform=False)
        bg_mask = get_mask(bg_shape, bg_im, deform=True)

        # # Only do the postprocess for the background mask
        bg_mask_postprocess = warp_mask(bg_mask, std=20)
        return fg_mask, bg_mask_postprocess

    
    def warp_images(self, fg_im, fg_shape, bg_im, bg_shape, fg_mask):
        """
        Warp foreground face onto background image using affine or 3D warping.
        """
        H, W, C = bg_im.shape
        use_3d_warp = np.random.rand() < 0.5

        if not use_3d_warp:
            aff_param = np.array(get_align_mat_new(fg_shape, bg_shape)).reshape(2, 3)
            warped_face = cv2.warpAffine(fg_im, aff_param, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            fg_mask = cv2.warpAffine(fg_mask, aff_param, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            fg_mask = fg_mask > 0
        else:
            warped_face = faceswap.warp_image_3d(fg_im, np.array(fg_shape[:48]), np.array(bg_shape[:48]), (H, W))
            fg_mask = np.mean(warped_face, axis=2) > 0

        return warped_face, fg_mask


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


    def blend_images(self, color_corrected_fg, bg_im, bg_mask, featherAmount=0.2):
        """
        Blend foreground and background images together.
        """
        # normalize the mask to have values between 0 and 1
        b_mask = bg_mask / 255.

        # Add an extra dimension and repeat the mask to match the number of channels in color_corrected_fg and bg_im
        b_mask = np.repeat(b_mask[:, :, np.newaxis], 3, axis=2)

        # Compute the alpha blending
        maskIndices = np.where(b_mask != 0)
        maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))

        # FIXME: deal with the bugs of empty maskpts
        if maskPts.size == 0:
            print(f"No non-zero values found in bg_mask for blending. Skipping this image.")
            return color_corrected_fg  # or handle this situation differently according to the needs

        faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
        featherAmount = featherAmount * np.max(faceSize)

        hull = cv2.convexHull(maskPts)
        dists = np.zeros(maskPts.shape[0])
        for i in range(maskPts.shape[0]):
            dists[i] = cv2.pointPolygonTest(hull, (int(maskPts[i, 0]), int(maskPts[i, 1])), True)

        weights = np.clip(dists / featherAmount, 0, 1)

        # Perform the blending operation
        color_corrected_fg = color_corrected_fg.astype(float)
        bg_im = bg_im.astype(float)
        blended_image = np.copy(bg_im)
        blended_image[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * color_corrected_fg[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * bg_im[maskIndices[0], maskIndices[1]]
        
        # Convert the blended image to 8-bit unsigned integers
        blended_image = np.clip(blended_image, 0, 255)
        blended_image = blended_image.astype(np.uint8)
        return blended_image


    def process_images(self, imid_fg, imid_bg, index):
        """
        Overview:
        Process foreground and background images following the data generation pipeline (BI dataset).

        Terminology:
        Foreground (fg) image: The image containing the face that will be blended onto the background image.
        Background (bg) image: The image onto which the face from the foreground image will be blended.
        """
        fg_im, fg_shape, bg_im, bg_shape = self.preprocess_images(imid_fg, imid_bg)
        fg_mask, bg_mask = self.generate_masks(fg_im, fg_shape, bg_im, bg_shape)
        warped_face, fg_mask = self.warp_images(fg_im, fg_shape, bg_im, bg_shape, fg_mask)

        try:
            # add the below two lines to make sure the bg_mask is strictly within the fg_mask
            bg_mask[fg_mask == 0] = 0
            color_corrected_fg = self.colorTransfer(bg_im, warped_face, bg_mask)
            blended_image = self.blend_images(color_corrected_fg, bg_im, bg_mask)
        # FIXME: ugly, in order to fix the problem of mask (all zero values for bg_mask)
        except:
            color_corrected_fg = self.colorTransfer(bg_im, warped_face, bg_mask)
            blended_image = self.blend_images(color_corrected_fg, bg_im, bg_mask)
        boundary = get_boundary(bg_mask)

        # # Prepare images and titles for the combined image
        # images = [fg_im, np.where(fg_mask>0, 255, 0), bg_im, bg_mask, color_corrected_fg, blended_image, np.where(boundary>0, 255, 0)]
        # titles = ["Fg Image", "Fg Mask", "Bg Image",
        #         "Bg Mask", "Blended Region",
        #         "Blended Image", "Boundary"]

        # # Save the combined image
        # os.makedirs('facexray_examples_3', exist_ok=True)
        # self.save_combined_image(images, titles, index, f'facexray_examples_3/combined_image_{index}.png')
        return blended_image, boundary, bg_im


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
        try:
            label = 1 if one_lmk_path.split('/')[6]=='manipulated_sequences' else 0
        except Exception as e:
            label = 1 if one_lmk_path.split('\\')[6] == 'manipulated_sequences' else 0
        imid_fg, imid_bg = self.get_fg_bg(one_lmk_path)
        manipulate_img, boundary, imid_bg = self.process_images(imid_fg, imid_bg, index)

        manipulate_img = self.post_proc(manipulate_img)
        imid_bg = self.post_proc(imid_bg)
        boundary = torch.from_numpy(boundary)
        boundary = boundary.unsqueeze(2).permute(2, 0, 1)

        # fake data
        fake_data_tuple = (manipulate_img, boundary, 1)
        # real data
        real_data_tuple = (imid_bg, torch.zeros_like(boundary), label)

        return fake_data_tuple, real_data_tuple


    @staticmethod
    def collate_fn(batch):
        """
        Collates batches of data and shuffles the images.
        """
        # Unzip the batch
        fake_data, real_data = zip(*batch)

        # Unzip the fake and real data
        fake_images, fake_boundaries, fake_labels = zip(*fake_data)
        real_images, real_boundaries, real_labels = zip(*real_data)

        # Combine fake and real data
        images = torch.stack(fake_images + real_images)
        boundaries = torch.stack(fake_boundaries + real_boundaries)
        labels = torch.tensor(fake_labels + real_labels)

        # Combine images, boundaries, and labels into tuples
        combined_data = list(zip(images, boundaries, labels))

        # Shuffle the combined data
        random.shuffle(combined_data)

        # Unzip the shuffled data
        images, boundaries, labels = zip(*combined_data)

        # Create the data dictionary
        data_dict = {
            'image': torch.stack(images),
            'label': torch.tensor(labels),
            'mask': torch.stack(boundaries),  # Assuming boundaries are your masks
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
