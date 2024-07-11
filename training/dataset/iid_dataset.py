'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is designed for scenarios such as disentanglement-based methods where it is necessary to ensure an equal number of positive and negative samples.
'''
import os.path
from copy import deepcopy
import cv2
import math
import torch
import random

import yaml
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import DataLoader

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

class IIDDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)


    def __getitem__(self, index):
        # Get the image paths and label
        image_path = self.data_dict['image'][index]
        if '\\' in image_path:
            per = image_path.split('\\')[-2]
        else:
            per = image_path.split('/')[-2]
        id_index = int(per.split('_')[-1])  # real video id
        label = self.data_dict['label'][index]

        # Load the image
        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            # Skip this image and return the first one
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)  # Convert to numpy array for data augmentation

        # Do Data Augmentation
        image_trans,_,_ = self.data_aug(image)

        # To tensor and normalize
        image_trans = self.normalize(self.to_tensor(image_trans))

        return id_index, image_trans, label

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        id_indexes, image_trans, label = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(image_trans, dim=0)
        labels = torch.LongTensor(label)
        ids = torch.LongTensor(id_indexes)
        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['id_index'] = ids
        data_dict['mask']=None
        data_dict['landmark']=None
        return data_dict


def draw_landmark(img,landmark):
    draw = ImageDraw.Draw(img)

    # landmark = np.stack([mean_face_x, mean_face_y], axis=1)
    # landmark *=256
    # 遍历每个特征点
    for i, point in enumerate(landmark):
        # 在图像上标记特征点
        draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=(255, 0, 0))
        # 在特征点旁边添加序号
        draw.text((point[0], point[1]), str(i), fill=(255, 255, 255))
    return img


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
    dataset = IIDDataset(config=config)
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=dataset.collate_fn)

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {batch}")

        # 如果数据集返回的是一个元组（例如，(data, target)），可以这样获取：
        img = batch['img']
