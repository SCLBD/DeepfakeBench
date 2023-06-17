import torch
import random
import numpy as np
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


class testDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='test'):
        super().__init__(config, mode)
        
        # Get real and fake image lists
        # Fix the label of real images to be 0 and fake images to be 1
        self.label_spe_list = []
        new_image_list = []
        for im_path in self.image_list:
            new_image_list.append(im_path)
            domain = im_path.split('/')[7]
            if domain == 'youtube':
                self.label_spe_list.append(0)
            elif domain == 'Deepfakes':
                self.label_spe_list.append(1)
            elif domain == 'Face2Face':
                self.label_spe_list.append(2)
            elif domain == 'FaceSwap':
                self.label_spe_list.append(3)
            elif domain == 'NeuralTextures':
                self.label_spe_list.append(4)
            elif domain == 'DeepFakeDetection':
                self.label_spe_list.append(6)  # real
            elif domain == 'actors':
                self.label_spe_list.append(5)  # fake
            elif domain == 'frames' and im_path.split('/')[5] == 'Celeb-DF-v2':
                if 'Celeb-real' in im_path:
                    self.label_spe_list.append(7)
                else:
                    self.label_spe_list.append(8)
            elif im_path.split('/')[4] == 'DFDCP':
                if im_path.split('/')[5] == 'original_videos':
                    self.label_spe_list.append(9)
                else:
                    self.label_spe_list.append(10)
            else:
                raise ValueError('Invalid domain {}'.format(domain))
        self.image_list = new_image_list

    def __getitem__(self, index):
        # Get the fake and real image paths and labels
        image_path = self.image_list[index]
        label_spe = self.label_spe_list[index]
        if label_spe == 0 or label_spe == 5 or label_spe == 7 or label_spe == 9:
            label = 0
        else:
            label = 1
        # label = 0 if label_spe == 0 else 1  # 0 for real, 1 for fake

        # Get the mask and landmark paths for fake and real images
        mask_path = image_path.replace('frames', 'masks')
        landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')

        # Load the fake and real images
        image = self.load_rgb(image_path)
        image = np.array(image)  # Convert to numpy array for data augmentation

        # Load mask and landmark (if needed) for fake and real images
        if self.config['with_mask']:
            mask = self.load_mask(mask_path)
        else:
            mask = None

        if self.config['with_landmark']:
            landmarks = self.load_landmark(landmark_path)
        else:
            landmarks = None

        # To tensor and normalize for fake and real images
        image = self.normalize(self.to_tensor(image))

        # Convert landmarks and masks to tensors if they exist
        if self.config['with_landmark']:
            landmarks = torch.from_numpy(landmarks)
        if self.config['with_mask']:
            mask = torch.from_numpy(mask)

        return image, label, label_spe, landmarks, mask

    def __len__(self):
        return len(self.image_list)

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
        # Separate the image, label, label_spe, landmark, and mask tensors
        images, labels, labels_spe, landmarks, masks = zip(*batch)
        
        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        labels_spe = torch.LongTensor(labels_spe)
        
        # Special case for landmarks and masks if they are None
        if landmarks[0] is not None:
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if masks[0] is not None:
            masks = torch.stack(masks, dim=0)
        else:
            masks = None
        
        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['label_spe'] = labels_spe
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        return data_dict


