'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2024-01-26

The code is designed for self-blending method (SBI, CVPR 2024).
'''

import sys
sys.path.append('.')

import os
import cv2
import yaml
import random
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from copy import deepcopy
import albumentations as A
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.sbi_api import SBI_API
from training.dataset.utils.bi_online_generation_yzy import random_get_hull
from training.dataset.SimSwap.test_one_image import self_blend

import warnings
warnings.filterwarnings('ignore')


from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_processor = SegformerImageProcessor.from_pretrained("/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/huggingface/hub/models--jonathandinu--face-parsing/snapshots/a2bf62f39dfd8f8856a3c19be8b0707a8d68abdd")
face_parser = SegformerForSemanticSegmentation.from_pretrained("/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/huggingface/hub/models--jonathandinu--face-parsing/snapshots/a2bf62f39dfd8f8856a3c19be8b0707a8d68abdd").to(device)


def create_facial_mask(mask, with_neck=False):
    facial_labels = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
    if with_neck:
        facial_labels += [17]
    facial_mask = np.zeros_like(mask, dtype=bool)
    for label in facial_labels:
        facial_mask |= (mask == label)
    return facial_mask.astype(np.uint8) * 255


def face_parsing_mask(img1, with_neck=False):
    # run inference on image
    img1 = Image.fromarray(img1)
    inputs = image_processor(images=img1, return_tensors="pt").to(device)
    outputs = face_parser(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

    # resize output to match input image dimensions
    upsampled_logits = nn.functional.interpolate(logits,
                    size=img1.size[::-1], # H x W
                    mode='bilinear',
                    align_corners=False)
    labels = upsampled_logits.argmax(dim=1)[0]
    mask = labels.cpu().numpy()
    mask = create_facial_mask(mask, with_neck)
    return mask


class YZYDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # Get real lists
        # Fix the label of real images to be 0
        self.real_imglist = [(img, label) for img, label in zip(self.image_list, self.label_list) if label == 0]


    def __getitem__(self, index):
        # Get the real image paths and labels
        real_image_path, real_label = self.real_imglist[index]
        # real_image_path = real_image_path.replace('/Youtu_Pangu_Security_Public/', '/Youtu_Pangu_Security/public/')

        # Load the real images
        real_image = self.load_rgb(real_image_path)
        real_image = np.array(real_image)  # Convert to numpy array

        # Face Parsing 
        mask = face_parsing_mask(real_image, with_neck=False)
        parse_mask_path = real_image_path.replace('frames', 'parse_mask')
        os.makedirs(os.path.dirname(parse_mask_path), exist_ok=True)
        cv2.imwrite(parse_mask_path, mask)

        # # SRI generation
        # sri_image = self_blend(real_image)
        # sri_path = real_image_path.replace('frames', 'sri_frames')
        # os.makedirs(os.path.dirname(sri_path), exist_ok=True)
        # cv2.imwrite(sri_path, sri_image)
        
    @staticmethod
    def collate_fn(batch):
        data_dict = {
            'image': None,
            'label': None,
            'landmark': None,
            'mask': None,
        }
        return data_dict

    def __len__(self):
        return len(self.real_imglist)



if __name__ == '__main__':
    with open('./training/config/detector/sbi.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config2['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = '/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/DeepfakeBenchv2/preprocessing/dataset_json'
    config.update(config2)
    train_set = YZYDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)