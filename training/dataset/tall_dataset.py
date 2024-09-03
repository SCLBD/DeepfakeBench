# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys

from torch import nn

sys.path.append('.')

import yaml
import numpy as np
from copy import deepcopy
import random
import torch
from torch.utils import data
from torchvision.utils import save_image
from training.dataset import DeepfakeAbstractBaseDataset
from einops import rearrange

FFpp_pool = ['FaceForensics++', 'FaceShifter', 'DeepFakeDetection', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']  #


def all_in_pool(inputs, pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class TALLDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """
        super().__init__(config, mode)

        assert self.video_level, "TALL is a videl-based method"
        assert int(self.clip_size ** 0.5) ** 2 == self.clip_size, 'clip_size must be square of an integer, e.g., 4'

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]  # for the image-level IO, only one frame is used

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2 ** 32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

            # Load the image
            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                print(f"Error loading image at index {index}: {e}")
                return self.__getitem__(0)
            image = np.array(image)  # Convert to numpy array for data augmentation

            # Load mask and landmark (if needed)
            if self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(landmarks)
                if self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:

            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)

            # cut out 16x16 patch
            F, C, H, W = image_tensors.shape
            x, y = np.random.randint(W), np.random.randint(H)
            x1 = np.clip(x - self.config['mask_grid_size'] // 2, 0, W)
            x2 = np.clip(x + self.config['mask_grid_size'] // 2, 0, W)
            y1 = np.clip(y - self.config['mask_grid_size'] // 2, 0, H)
            y2 = np.clip(y + self.config['mask_grid_size'] // 2, 0, H)
            image_tensors[:, :, y1:y2, x1:x2] = -1

            # # concatenate sub-image and reszie to 224x224
            # image_tensors = image_tensors.reshape(-1, H, W)
            # image_tensors = rearrange(image_tensors, '(rh rw c) h w -> c (rh h) (rw w)', rh=2, c=C)
            # image_tensors = nn.functional.interpolate(image_tensors.unsqueeze(0),
            #                                           size=(self.config['resolution'], self.config['resolution']),
            #                                           mode='bilinear', align_corners=False).squeeze(0)
            # Stack landmark and mask tensors along a new dimension (time)
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in
                       landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]
            # Get the first landmark and mask tensors
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in
                       landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]

        return image_tensors, label, landmark_tensors, mask_tensors


if __name__ == "__main__":
    with open('training/config/detector/tall.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = TALLDataset(
        config=config,
        mode='train',
    )
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
        print(batch['image'].shape)
        print(batch['label'])
        b, f, c, h, w = batch['image'].shape
        for i in range(f):
            img_tensor = batch['image'][0][i]
            img_tensor = img_tensor * torch.tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1) + torch.tensor(
                [0.5, 0.5, 0.5]).reshape(-1, 1, 1)
            save_image(img_tensor, f'{i}.png')

        break
