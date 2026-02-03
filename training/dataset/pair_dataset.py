'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is designed for scenarios such as disentanglement-based methods where it is necessary to ensure an equal number of positive and negative samples.
'''

import torch
import random
import numpy as np
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


class pairDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # Get real and fake image lists
        # Fix the label of real images to be 0 and fake images to be 1
        self.fake_imglist = [(img, label, 1) for img, label in zip(self.image_list, self.label_list) if label != 0]
        self.real_imglist = [(img, label, 0) for img, label in zip(self.image_list, self.label_list) if label == 0]

    def __getitem__(self, index, norm=True):
        # Get the fake and real image paths and labels
        fake_image_path, fake_spe_label, fake_label = self.fake_imglist[index]
        real_index = random.randint(0, len(self.real_imglist) - 1)  # Randomly select a real image
        real_image_path, real_spe_label, real_label = self.real_imglist[real_index]

        # Get the mask and landmark paths for fake and real images
        fake_mask_path = fake_image_path.replace('frames', 'masks')
        fake_landmark_path = fake_image_path.replace('frames', 'landmarks').replace('.png', '.npy')
        
        real_mask_path = real_image_path.replace('frames', 'masks')
        real_landmark_path = real_image_path.replace('frames', 'landmarks').replace('.png', '.npy')

        # Load the fake and real images
        fake_image = self.load_rgb(fake_image_path)
        real_image = self.load_rgb(real_image_path)

        fake_image = np.array(fake_image)  # Convert to numpy array for data augmentation
        real_image = np.array(real_image)  # Convert to numpy array for data augmentation

        # Load mask and landmark (if needed) for fake and real images
        if self.config['with_mask']:
            fake_mask = self.load_mask(fake_mask_path)
            real_mask = self.load_mask(real_mask_path)
        else:
            fake_mask, real_mask = None, None

        if self.config['with_landmark']:
            fake_landmarks = self.load_landmark(fake_landmark_path)
            real_landmarks = self.load_landmark(real_landmark_path)
        else:
            fake_landmarks, real_landmarks = None, None

        # Do transforms for fake and real images
        fake_image_trans, fake_landmarks_trans, fake_mask_trans = self.data_aug(fake_image, fake_landmarks, fake_mask)
        real_image_trans, real_landmarks_trans, real_mask_trans = self.data_aug(real_image, real_landmarks, real_mask)

        if not norm:
            return {"fake": (fake_image_trans, fake_label), 
                    "real": (real_image_trans, real_label)}

        # To tensor and normalize for fake and real images
        fake_image_trans = self.normalize(self.to_tensor(fake_image_trans))
        real_image_trans = self.normalize(self.to_tensor(real_image_trans))

        # Convert landmarks and masks to tensors if they exist
        if self.config['with_landmark']:
            fake_landmarks_trans = torch.from_numpy(fake_landmarks_trans)
            real_landmarks_trans = torch.from_numpy(real_landmarks_trans)
        if self.config['with_mask']:
            fake_mask_trans = torch.from_numpy(fake_mask_trans)
            real_mask_trans = torch.from_numpy(real_mask_trans)

        return {"fake": (fake_image_trans, fake_label, fake_spe_label, fake_landmarks_trans, fake_mask_trans), 
                "real": (real_image_trans, real_label, real_spe_label, real_landmarks_trans, real_mask_trans)}

    def __len__(self):
        return len(self.fake_imglist)

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
        # Separate the image, label, landmark, and mask tensors for fake and real data
        fake_images, fake_labels, fake_spe_labels, fake_landmarks, fake_masks = zip(*[data["fake"] for data in batch])
        real_images, real_labels, real_spe_labels, real_landmarks, real_masks = zip(*[data["real"] for data in batch])

        # Stack the image, label, landmark, and mask tensors for fake and real data
        fake_images = torch.stack(fake_images, dim=0)
        fake_labels = torch.LongTensor(fake_labels)
        fake_spe_labels = torch.LongTensor(fake_spe_labels)
        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)
        real_spe_labels = torch.LongTensor(real_spe_labels)

        # Special case for landmarks and masks if they are None
        if fake_landmarks[0] is not None:
            fake_landmarks = torch.stack(fake_landmarks, dim=0)
        else:
            fake_landmarks = None
        if real_landmarks[0] is not None:
            real_landmarks = torch.stack(real_landmarks, dim=0)
        else:
            real_landmarks = None

        if fake_masks[0] is not None:
            fake_masks = torch.stack(fake_masks, dim=0)
        else:
            fake_masks = None
        if real_masks[0] is not None:
            real_masks = torch.stack(real_masks, dim=0)
        else:
            real_masks = None

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        spe_labels = torch.cat([real_spe_labels, fake_spe_labels], dim=0)
        
        if fake_landmarks is not None and real_landmarks is not None:
            landmarks = torch.cat([real_landmarks, fake_landmarks], dim=0)
        else:
            landmarks = None

        if fake_masks is not None and real_masks is not None:
            masks = torch.cat([real_masks, fake_masks], dim=0)
        else:
            masks = None

        data_dict = {
            'image': images,
            'label': labels,
            'label_spe': spe_labels,
            'landmark': landmarks,
            'mask': masks
        }
        return data_dict

