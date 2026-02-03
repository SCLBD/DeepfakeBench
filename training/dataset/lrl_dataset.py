import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import cv2
import random
import yaml
import torch
import numpy as np
from copy import deepcopy
import albumentations as A
from .abstract_dataset import DeepfakeAbstractBaseDataset
from PIL import Image

c=0

class LRLDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        global c
        c=config

    def multi_pass_filter(self, img, r1=0.33, r2=0.66):
        rows, cols = img.shape
        k = cols / rows

        mask = np.zeros((rows, cols), np.uint8)
        x, y = np.ogrid[:rows, :cols]
        mask_area = (k * x + y < r1 * cols)
        mask[mask_area] = 1
        low_mask = mask

        mask = np.ones((rows, cols), np.uint8)
        x, y = np.ogrid[:rows, :cols]
        mask_area = (k * x + y < r2 * cols)
        mask[mask_area] = 0
        high_mask = mask

        mask1 = np.zeros((rows, cols), np.uint8)
        mask1[low_mask == 0] = 1
        mask2 = np.zeros((rows, cols), np.uint8)
        mask2[high_mask == 0] = 1
        mid_mask = mask1 * mask2

        return low_mask, mid_mask, high_mask
    
    def image2dct(self,img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = np.float32(img_gray)
        img_dct = cv2.dct(img_gray)
        # img_dct = np.log(np.abs(img_dct)+1e-6)

        low_mask, mid_mask, high_mask = self.multi_pass_filter(img_dct, r1=0.33, r2=0.33)
        img_dct_filterd = high_mask * img_dct
        img_idct = cv2.idct(img_dct_filterd)

        return img_idct

    def __getitem__(self, index):
        image_trans, label, landmark_tensors, mask_trans = super().__getitem__(index, no_norm=True)

        img_idct = self.image2dct(image_trans)
        # normalize idct
        img_idct = (img_idct / 255 - 0.5) / 0.5
        # img_idct = img_idct[np.newaxis, ...]

        # To tensor and normalize for fake and real images
        image_trans = self.normalize(self.to_tensor(image_trans))
        img_idct_trans = self.to_tensor(img_idct)
        mask_trans = torch.from_numpy(mask_trans)
        mask_trans = mask_trans.squeeze(2).permute(2, 0, 1)
        mask_trans = torch.mean(mask_trans, dim=0, keepdim=True)
        return image_trans, label, img_idct_trans, mask_trans

    def __len__(self):
        return len(self.image_list)


    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor and label tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        global c
        images, labels, img_idct_trans, masks = zip(*batch)
        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        masks = torch.stack(masks, dim=0)
        img_idct_trans = torch.stack(img_idct_trans, dim=0)

        data_dict = {
            'image': images,
            'label': labels,
            'landmark': None,
            'idct': img_idct_trans,
            'mask': masks,
        }
        return data_dict



if __name__ == '__main__':
    with open(r'H:\code\DeepfakeBench\training\config\detector\lrl_effnb4.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(r'H:\code\DeepfakeBench\training\config\train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])
    config2['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config.update(config2)
    train_set = LRLDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        if iteration > 10:
            break