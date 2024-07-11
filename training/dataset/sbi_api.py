# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset,IterableDataset
from glob import glob
import os
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
import scipy as sp
from skimage.measure import label, regionprops
from training.dataset.library.bi_online_generation import random_get_hull
import albumentations as alb

import warnings
warnings.filterwarnings('ignore')


def alpha_blend(source,target,mask):
	mask_blured = get_blend_mask(mask)
	img_blended=(mask_blured * source + (1 - mask_blured) * target)
	return img_blended,mask_blured


def dynamic_blend(source,target,mask):
	mask_blured = get_blend_mask(mask)
	blend_list=[0.25,0.5,0.75,1,1,1]
	blend_ratio = blend_list[np.random.randint(len(blend_list))]
	mask_blured*=blend_ratio
	img_blended=(mask_blured * source + (1 - mask_blured) * target)
	return img_blended,mask_blured


def get_blend_mask(mask):
	H,W=mask.shape
	size_h=np.random.randint(192,257)
	size_w=np.random.randint(192,257)
	mask=cv2.resize(mask,(size_w,size_h))
	kernel_1=random.randrange(5,26,2)
	kernel_1=(kernel_1,kernel_1)
	kernel_2=random.randrange(5,26,2)
	kernel_2=(kernel_2,kernel_2)
	
	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured[mask_blured<1]=0
	
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured = cv2.resize(mask_blured,(W,H))
	return mask_blured.reshape((mask_blured.shape+(1,)))


def get_alpha_blend_mask(mask):
	kernel_list=[(11,11),(9,9),(7,7),(5,5),(3,3)]
	blend_list=[0.25,0.5,0.75]
	kernel_idxs=random.choices(range(len(kernel_list)), k=2)
	blend_ratio = blend_list[random.sample(range(len(blend_list)), 1)[0]]
	mask_blured = cv2.GaussianBlur(mask, kernel_list[0], 0)
	# print(mask_blured.max())
	mask_blured[mask_blured<mask_blured.max()]=0
	mask_blured[mask_blured>0]=1
	# mask_blured = mask
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_list[kernel_idxs[1]], 0)
	mask_blured = mask_blured/(mask_blured.max())
	return mask_blured.reshape((mask_blured.shape+(1,)))


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
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
	


def get_boundary(mask, apply_dilation=True, apply_motion_blur=True):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    if mask.max() > 1:
        boundary = mask / 255.
    else:
        boundary = mask
    boundary = 4 * boundary * (1. - boundary)

    boundary = boundary * 255
    boundary = random_dilate(boundary)

    if apply_motion_blur:
        boundary = random_motion_blur(boundary)
        boundary = boundary / 255.
    return boundary

def random_dilate(mask, max_kernel_size=5):
    kernel_size = random.randint(1, max_kernel_size)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

def random_motion_blur(mask, max_kernel_size=5):
    kernel_size = random.randint(1, max_kernel_size)
    kernel = np.zeros((kernel_size, kernel_size))
    anchor = random.randint(0, kernel_size - 1)
    kernel[:, anchor] = 1 / kernel_size
    motion_blurred_mask = cv2.filter2D(mask, -1, kernel)
    return motion_blurred_mask



class SBI_API:
	def __init__(self,phase='train',image_size=256):
		
		assert phase == 'train', f"Current SBI API only support train phase, but got {phase}"

		self.image_size=(image_size,image_size)
		self.phase=phase

		self.transforms=self.get_transforms()
		self.source_transforms = self.get_source_transforms()
		self.bob_transforms = self.get_source_transforms_for_bob()


	def __call__(self,img,landmark=None):
		try:  
			assert landmark is not None, "landmark of the facial image should not be None."  
			# img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy())  
			
			if random.random() < 1.0:
				# apply sbi
				img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy())
			else:
				# apply boundary motion blur (bob)
				img_r,img_f,mask_f=self.bob(img.copy(),landmark.copy())
			
			if self.phase=='train':  
				transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))  
				img_f=transformed['image']  
				img_r=transformed['image1']  
			return img_f,img_r  
		except Exception as e:  
			print(e)  
			return None,None
		
    
	def get_source_transforms(self):
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)

		
	def get_transforms(self):
		return alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def randaffine(self,img,mask):
		f=alb.Affine(
				translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
				scale=[0.95,1/0.95],
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=50,
				sigma=7,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask
	

	def get_source_transforms_for_bob(self):
		return alb.Compose([
				alb.Compose([
						alb.ImageCompression(quality_lower=40,quality_upper=100,p=1),
					],p=1),

				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),

			], p=1.)

	def bob(self,img,landmark):
		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		# mask=np.zeros_like(img[:,:,0])
		# cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)
		hull_type = random.choice([0, 1, 2, 3])
		mask=random_get_hull(landmark,img,hull_type)[:,:,0]

		source = img.copy()
		source = self.bob_transforms(image=source.astype(np.uint8))['image']
		source, mask = self.randaffine(source,mask)
		mask = get_blend_mask(mask)

		# get boundary with motion blur
		boundary = get_boundary(mask)

		blend_list = [0.25,0.5,0.75,1,1,1]
		blend_ratio = blend_list[np.random.randint(len(blend_list))]
		boundary *= blend_ratio
		boundary = np.repeat(boundary[:, :, np.newaxis], 3, axis=2)
		img_blended = (boundary * source + (1 - boundary) * img)

		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,boundary.squeeze()

		
	def self_blending(self,img,landmark):
		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		# mask=np.zeros_like(img[:,:,0])
		# cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)
		hull_type = random.choice([0, 1, 2, 3])
		mask=random_get_hull(landmark,img,hull_type)[:,:,0]

		source = img.copy()
		if np.random.rand()<0.5:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.randaffine(source,mask)

		img_blended,mask=dynamic_blend(source,img,mask)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	

	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark


	def hflip(self,img,mask=None,landmark=None,bbox=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		if bbox is not None:
			bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)

			
			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]
			
		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None
		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new
	

if __name__=='__main__':
	seed=10
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	api=SBI_API(phase='train',image_size=256)
	
	img_path = 'FaceForensics++/original_sequences/youtube/c23/frames/000/000.png'
	img = cv2.imread(img_path)
	landmark_path = img_path.replace('frames', 'landmarks').replace('png', 'npy')
	landmark = np.load(landmark_path)
	sbi_img, ori_img = api(img, landmark)
