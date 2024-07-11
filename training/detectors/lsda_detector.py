'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the LSDADetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@article{yan2023transcending,
  title={Transcending forgery specificity with latent space augmentation for generalizable deepfake detection},
  author={Yan, Zhiyuan and Luo, Yuhao and Lyu, Siwei and Liu, Qingshan and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2311.11278},
  year={2023}
}
'''


import os
import datetime
import numpy as np
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
import cv2
from collections import defaultdict


from efficientnet_pytorch import EfficientNet
from networks.iresnet import iresnet100
from networks.xception import Xception
from detectors import DETECTOR
from sklearn import metrics
from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC


device = "cuda" if torch.cuda.is_available() else "cpu"



@DETECTOR.register_module(module_name='lsda')
class LSDADetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        # model
        forgery_num = 4
        self.model = generator(
            num_classes=forgery_num+1, encoder_feat_dim=512, 
            teacher=config['teacher'], student=config['student'],
            real_encoder=config['real_encoder'],
        ).to(device)

        # loss
        self.cls_criterion = nn.CrossEntropyLoss()
        self.gan_loss_fn = nn.BCELoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        pass  # FIXME: will be added into this function

    def build_loss(self, config):
        pass  # FIXME: will be added into this function

    def features(self, data_dict: dict) -> torch.tensor:
        pass  # FIXME: will be added into this function

    def classifier(self, features: torch.tensor) -> torch.tensor:
        pass  # FIXME: will be added into this function

    def get_losses(self, data_dict: dict, predictions: dict) -> dict:
        try:
            deepfake_loss, total_loss_distillation, domain_loss, loss_real = predictions['pred_loss']

            loss = \
                1  * domain_loss + \
                0.5  * deepfake_loss + \
                1  * total_loss_distillation + \
                1  * loss_real
            loss_dict = {'overall': loss, 'domain': domain_loss, 'deepfake': deepfake_loss, 'distillation': total_loss_distillation, 'real_loss': loss_real}
        
        except:
            # test time
            loss = 0
            loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        label = torch.where(label == 0, 0, 1).reshape(-1,1)
        prob = pred_dict['prob'].reshape(-1,1)
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), prob.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        
        # 1. Forward pass
        # pred, data_dict['label'], feat = 
        model_output = self.model(data_dict['image'], data_dict['label'], inference=inference)
        if inference:
            pred = model_output
            prob = torch.softmax(pred, dim=1)[:, 1]
            pred_dict = {'cls': pred, 'prob': prob, 'feat': prob}
        else:
            pred, deepfake_loss, total_loss_distillation, domain_loss, loss_real, student_feature = model_output
            loss = (deepfake_loss, total_loss_distillation, domain_loss, loss_real)
            prob = torch.softmax(pred, dim=1)[:, 1]
            pred_dict = {'cls': pred, 'prob': prob, 'feat': student_feature, 'pred_loss': loss}

        if inference:
            self.prob.append(
                pred_dict['prob']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(pred, 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)

        return pred_dict


class efficientnet(nn.Module):
    def __init__(self, pretrain='efficientnet-b4', sbi=None):
        super(efficientnet, self).__init__()
        self.model = EfficientNet.from_pretrained(pretrain,weights_path='./training/pretrained/efficientnet-b4-6ed6700e.pth')

        if pretrain == 'efficientnet-b4':
            self.conv = nn.Conv2d(1792, 512, 1)
        elif pretrain == 'efficientnet-b1':
            self.conv = nn.Conv2d(1280, 512, 1)
        elif pretrain == 'efficientnet-b3':
            self.conv = nn.Conv2d(1536, 512, 1)
        elif pretrain == 'efficientnet-b5':
            self.conv = nn.Conv2d(2048, 512, 1)
        elif pretrain == 'efficientnet-b6':
            self.conv = nn.Conv2d(2304, 512, 1)
        else:
            raise ValueError('pretrain is not supported')

        # self.channel_adjust_conv = nn.Conv2d(2424, 512, 1)
    
    def features(self, x):
        x = self.model.extract_features(x)
        x = self.conv(x)

        return x
    
    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.conv(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(MLP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        x = self.pool(x)
        x = self.mlp(x)
        return x

class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(hidden_dim, out_f, 1, 1),)

    def forward(self, x):
        x = self.conv2d(x)
        return x

class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Head, self).__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        bs = x.size()[0]
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class generator(nn.Module):
    def __init__(self, num_classes, 
                 encoder_feat_dim, 
                 num_domains=5, 
                 teacher='efficientnetb4', 
                 student='efficientnetb4',
                 real_encoder=None,
        ) -> None:

        super(generator, self).__init__()
        self.num_domains = num_domains
        # init variable
        self.num_classes = num_classes
        self.encoder_feat_dim = encoder_feat_dim
        self.half_fingerprint_dim = encoder_feat_dim//2
        
        # basic function
        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.count = 0

        # 4个fake，用modulelist
        if teacher == 'xception':
            self.encoders_f = nn.ModuleList([self.init_xcep() for _ in range(self.num_domains-1)])
        elif teacher == 'efficientnetb4':
            self.encoders_f = nn.ModuleList([self.init_efficient() for _ in range(self.num_domains-1)])
        
        if real_encoder is None:
            self.encoder_c = iresnet100(pretrained=False, fp16=False)
        elif real_encoder == 'efficientnetb4':
            print('real encoder: efficient')
            self.encoder_c = self.init_efficient()
            

        if student == 'xception':
            self.student_encoder = self.init_xcep()
        elif student == 'efficientnetb4':
            self.student_encoder = self.init_efficient()

        self.fc_weights = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            )

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.half_fingerprint_dim*2, self.half_fingerprint_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.half_fingerprint_dim, num_domains),
        )

        self.binary_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.encoder_feat_dim, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 2),
        )

        self.cls_criterion = nn.CrossEntropyLoss()

    def init_xcep(self, pretrained_path='pretrained/xception-b5690688.pth'):
        xcep = Xception(self.num_classes)
        # load pre-trained Xception
        state_dict = torch.load(pretrained_path)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        xcep.load_state_dict(state_dict, False)
        return xcep

    def init_efficient(self):
        model = efficientnet(pretrain='efficientnet-b4')
        return model

    # only for grad cam
    def features(self, cat_data):
        # Binary classification detector, a student model, to be distilled (real/fake)
        student_feature = self.student_encoder.features(cat_data)
        return student_feature

    # only for grad cam
    def classifier(self, fea):
        out = self.binary_classifier(fea)
        return out, None

    def real_fake_feature_extract(self, cat_data):
        number_of_groups, video_per_group, c, h, w = cat_data.shape

        # Use defaultdict to store tensors for each domain
        domain_f_chunks = defaultdict(list)
        domain_c_chunks = defaultdict(list)

        for domain_id in range(video_per_group):
            # Get the data for the current domain across all groups
            domain_data_tensor = cat_data[:, domain_id]
            
            # Compute self-generation loss
            c_chunk = self.encoder_c(domain_data_tensor)
            if domain_id>0: # 是哪个domain的。就用哪个encoder. 5 in total
                f_chunk = self.encoders_f[domain_id-1].features(domain_data_tensor)
                # Store the chunks in the defaultdict
                domain_f_chunks[domain_id-1] = f_chunk
            domain_c_chunks[domain_id] = c_chunk

        # Reconstruct the tensors based on the label order
        all_f_outputs = torch.stack(list(domain_f_chunks.values())).transpose(1, 0)
        all_c_outputs = torch.stack(list(domain_c_chunks.values())).transpose(1, 0)

        return all_f_outputs, all_c_outputs


    def augment_domains(self, groups_feature_maps):
        # Helper Functions
        def hard_example_interpolation(z_i, hard_example, lambda_1):
            return z_i + lambda_1 * (hard_example - z_i)

        def hard_example_extrapolation(z_i, mean_latent, lambda_2):
            return z_i + lambda_2 * (z_i - mean_latent)

        def add_gaussian_noise(z_i, sigma, lambda_3):
            epsilon = torch.randn_like(z_i) * sigma
            return z_i + lambda_3 * epsilon

        def difference_transform(z_i, z_j, z_k, lambda_4):
            return z_i + lambda_4 * (z_j - z_k)

        def distance(z_i, z_j):
            return torch.norm(z_i - z_j)


        domain_number = len(groups_feature_maps[0])

        # Calculate the mean latent vector for each domain across all groups; why 8*8
        domain_means = []
        for domain_idx in range(domain_number):
            all_samples_in_domain = torch.cat([group[domain_idx] for group in groups_feature_maps], dim=0)
            domain_mean = torch.mean(all_samples_in_domain, dim=0)
            domain_means.append(domain_mean)

        # Identify the hard example for each domain across all groups (the farest one)
        hard_examples = []
        for domain_idx in range(domain_number):
            all_samples_in_domain = torch.cat([group[domain_idx] for group in groups_feature_maps], dim=0)
            distances = torch.tensor([distance(z, domain_means[domain_idx]) for z in all_samples_in_domain])
            hard_example = all_samples_in_domain[torch.argmax(distances)]
            hard_examples.append(hard_example)


        augmented_groups = []
        # modify each feature maps
        for group_feature_maps in groups_feature_maps:
            augmented_domains = []

            for domain_idx, domain_feature_maps in enumerate(group_feature_maps):
                # Choose a random augmentation
                augmentations = [
                    lambda z: hard_example_interpolation(z, hard_examples[domain_idx], random.random()),
                    lambda z: hard_example_extrapolation(z, domain_means[domain_idx], random.random()),
                    lambda z: add_gaussian_noise(z, random.random(), random.random()),
                    lambda z: difference_transform(z, domain_feature_maps[0], domain_feature_maps[1], random.random())
                ]
                chosen_aug = random.choice(augmentations)
                augmented = torch.stack([chosen_aug(z) for z in domain_feature_maps])
                augmented_domains.append(augmented)

            augmented_domains = torch.stack(augmented_domains)
            augmented_groups.append(augmented_domains)

        return torch.stack(augmented_groups)


    def mixup_in_latent_space(self, data):
        # data shape: [batchsize, num_domains, 3, 256, 256]
        bs, num_domains, _, _, _ = data.shape

        # Initialize an empty tensor for mixed data
        mixed_data = torch.zeros_like(data)

        # For each sample in the batch
        for i in range(bs):
            # Step 1: Generate a shuffled index list for the domains
            shuffled_idxs = torch.randperm(num_domains)

            # Step 2: Choose random alpha between 0.5 and 2, then sample lambda from beta distribution
            alpha = torch.rand(1) * 1.5 + 0.5  # random alpha between 0.5 and 2
            lambda_ = torch.distributions.beta.Beta(alpha, alpha).sample().to(data.device)

            # Step 3: Perform mixup using the shuffled indices
            mixed_data[i] = lambda_ * data[i] + (1 - lambda_) * data[i, shuffled_idxs]

        return mixed_data


    def rotate_trans(self, fake_fs, 
                    rotation_degree_range=(-30, 30)):
    
        # Convert degrees to radians
        rotation_degree = torch.rand(1).to(fake_fs.device) * (rotation_degree_range[1] - rotation_degree_range[0]) + rotation_degree_range[0]
        rotation_radians = rotation_degree * (3.141592653589793 / 180.0)
        # Create an identity affine transformation (3x4) with the rotation in the top-left 2x2 corner
        identity_affine = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=torch.float32).to(fake_fs.device)
        # Fill the rotation into the top-left 2x2
        identity_affine[0, 0:2] = torch.tensor([torch.cos(rotation_radians), -torch.sin(rotation_radians)], dtype=torch.float32).to(fake_fs.device)
        identity_affine[1, 0:2] = torch.tensor([torch.sin(rotation_radians), torch.cos(rotation_radians)], dtype=torch.float32).to(fake_fs.device)
        # Expand the affine transformation for the batch
        theta = identity_affine.unsqueeze(0).repeat(fake_fs.size(0), 1, 1)
        grid = F.affine_grid(theta, fake_fs.size())
        fake_fs = F.grid_sample(fake_fs, grid)
    
        return fake_fs


    @staticmethod
    def cosine_similarity_loss(x, y, dim=1, eps=1e-8):
        x_norm = x / (x.norm(dim=dim, keepdim=True) + eps)
        y_norm = y / (y.norm(dim=dim, keepdim=True) + eps)
        cos_sim = (x_norm * y_norm).sum(dim=dim)
        return 1 - cos_sim
    

    @staticmethod
    def js_loss(inputs, targets):
        """
        Computes the Jensen-Shannon divergence loss.
        """
        # Compute the probability distributions
        inputs_prob = F.softmax(inputs, dim=1)
        targets_prob = F.softmax(targets, dim=1)

        # Compute the average probability distribution
        avg_prob = (inputs_prob + targets_prob) / 2

        # Compute the KL divergence component for each distribution
        kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        kl_inputs = kl_div_loss(inputs_prob.log(), avg_prob)
        kl_targets = kl_div_loss(targets_prob.log(), avg_prob)

        # Compute the Jensen-Shannon divergence
        loss = 0.5 * (kl_inputs + kl_targets)
        return loss


    def forward(self, cat_data, label=None, inference=False):
        if inference:
            # Use the common encoder for inference/testing
            student_feature = self.student_encoder.features(cat_data) 
            out_common = self.binary_classifier(student_feature)
            return out_common

        # Obtain data
        number_of_groups, video_per_group, c, h, w = cat_data.shape

        # Extract the real and fake features separately ; 每一个都是一个单独的effnb4提取出来的
        f_outputs, c_outputs = self.real_fake_feature_extract(cat_data)

        # p = random.random()
        # if p > 0.5:
        #     f_outputs = self.rotate_trans(f_outputs)



        # Perform augmentation in the latent space / f_out 只包含 fake
        f_outputs_aug = self.augment_domains(f_outputs)
        # Mixup in the latent space for cross-domain
        mix_f_outputs = self.mixup_in_latent_space(f_outputs)
        aug_fake = torch.cat([f_outputs_aug, mix_f_outputs], dim=2).view(-1, self.encoder_feat_dim*2, 8, 8)
        fc = self.fc_weights(aug_fake).view(number_of_groups, video_per_group-1, self.encoder_feat_dim, 8, 8)



        # real constrain (optional, for the aim of learning real-features (e.g., ID) better)
        real = c_outputs[:, 0, :, :, :]
        df = c_outputs[:, 1, :, :, :]
        f2f = c_outputs[:, 2, :, :, :]
        fs = c_outputs[:, 3, :, :, :]
        nt = c_outputs[:, 4, :, :, :]
        loss_real = self.cosine_similarity_loss(real, nt).sum() \
          + self.cosine_similarity_loss(real, f2f).sum() \
          - self.cosine_similarity_loss(real, fs).sum() \
          - self.cosine_similarity_loss(real, df).sum()
        # loss_real = self.js_loss(real, nt) + self.js_loss(real, f2f) - self.js_loss(real, fs) - self.js_loss(real, df)
        loss_real = loss_real.mean()
                    



        # Obtain reshape label
        label = label.contiguous().view(-1)
        # Obtain the binary label
        binary_label = torch.where(label==0, 0, 1)




        # Binary classification detector, a student model, to be distilled (real/fake)
        student_feature = self.student_encoder.features(cat_data.view(-1, c, h, w))
        binary_out = self.binary_classifier(student_feature)
        deepfake_loss = F.cross_entropy(binary_out, binary_label)


        # Distillation for the student encoder
        real_mask = (label == 0)
        fake_mask = (label > 0)
        distillation_real_feature = student_feature[real_mask]
        distillation_fake_feature = student_feature[fake_mask].reshape((number_of_groups, video_per_group-1, self.encoder_feat_dim, 8, 8))
        loss_distillation_real = F.mse_loss(distillation_real_feature, c_outputs.reshape(((-1, self.encoder_feat_dim, 8, 8)))[real_mask])
        loss_distillation_fake = F.mse_loss(distillation_fake_feature, fc)
        total_loss_distillation = loss_distillation_real + loss_distillation_fake




        # Domain classification loss for all domains
        all_domain_feat = torch.cat([c_outputs[:, 0, :, :, :].unsqueeze(1), f_outputs], dim=1).reshape((number_of_groups*video_per_group, self.encoder_feat_dim, 8, 8))
        out_spe = self.mlp(all_domain_feat)
        domain_loss = self.cls_criterion(out_spe, label)

        return binary_out, deepfake_loss, total_loss_distillation, domain_loss, loss_real, student_feature
