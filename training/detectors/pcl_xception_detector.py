'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the PCLDetector

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
@inproceedings{zhao2021learning,
  title={Learning self-consistency for deepfake detection},
  author={Zhao, Tianchen and Xu, Xiang and Xu, Mingze and Ding, Hui and Xiong, Yuanjun and Xia, Wei},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={15023--15033},
  year={2021}
}
'''


import os
import datetime
import logging
import random

import numpy as np
import yaml
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from dataset.I2G_dataset import I2GDataset
from metrics.base_metrics_class import calculate_metrics_for_train

from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import math
from torchvision import transforms

logger = logging.getLogger(__name__)


class Masks4D(object):
    def __call__(self, masks):

        first_w = True
        first_h = True
        first_c = True

        for k, mask in enumerate(masks):
            mask=mask.squeeze(0)
            h, w = mask.shape
            real_mask = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mask, 0), 0), 0)
            # fake_mask = torch.unsqueeze(torch.unsqueeze(1 - mask, 0), 0)
            for i, mask_h in enumerate(mask):
                for j, mask_w in enumerate(mask_h):
                    curr_mask = 1 - torch.abs(mask_w - real_mask)
                    if first_w:
                        total_mask_w = real_mask
                        first_w = False
                    else:
                        total_mask_w = torch.cat((total_mask_w, curr_mask), dim=2)
                if first_h:
                    total_mask_h = total_mask_w
                    first_h = False
                else:
                    total_mask_h = torch.cat((total_mask_h, total_mask_w), dim = 1)
                first_w = True
            if first_c:
                total_mask_c = total_mask_h
                first_c = False
            else:
                total_mask_c = torch.cat((total_mask_c, total_mask_h), dim = 0)
            first_h = True
        return total_mask_c


class NLBlockND(nn.Module):
    def __init__(self, in_channels=256):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        self.in_channels = in_channels

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions

        # add BatchNorm layer after the last conv layer
        self.sig = nn.Sigmoid()

        # define theta and phi for all operations except gaussian 为什么会有俩？
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

    def forward(self, x, return_nl_map=False):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation

        theta_x = self.theta(x).view(batch_size, self.in_channels, -1) #flatten operation
        #phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        phi_x = self.theta(x).view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        #channel as vector
        f = torch.matmul(theta_x, phi_x)
        f_div_C = f / math.sqrt(self.in_channels)

        # contiguous here just allocates contiguous chunk of memory
        y = f_div_C.permute(0, 2, 1).contiguous()

        sig_y = self.sig(y)
        final_y = sig_y.view(batch_size, *x.size()[2:], *x.size()[2:])

        if return_nl_map:
            return final_y, sig_y
        else:
            return final_y


@DETECTOR.register_module(module_name='pcl_xception')
class PCLXceptionDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.PCL = NLBlockND(in_channels=728)
        self.Msk_PCL = transforms.Compose([Masks4D()])
        self.mask_down_sampling = nn.UpsamplingBilinear2d(
            scale_factor=16 / 256)
        self.criterionBCE = nn.BCELoss()

    def build_backbone(self, config):
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        if config['pretrained'] != 'None':
            # if donot load the pretrained weights, fail to get good results
            state_dict = torch.load(config['pretrained'])
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
            # backbone.classifier=classifier
            backbone.load_state_dict(state_dict, False)
            logger.info('Load pretrained model successfully!')
        else:
            logger.info('No pretrained model.')
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        if config['loss_func']=='center_loss':
            loss_func = loss_class(num_classes=2, feat_dim=2048)
        else:
            loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        if pred_dict['pcl_map'] is not None:
            pcl_loss = self.criterionBCE(pred_dict['pcl_map'],pred_dict['pcl_gt_map'])
        else:
            pcl_loss = 0
        det_loss = self.loss_func(pred, label)
        loss = det_loss+ self.config['pcl_loss_weight'] * pcl_loss
        loss_dict = {'overall': loss,'pcl_loss': pcl_loss, 'det_loss':det_loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        #print(data_dict['image'].device)
        # get the features by backbone
        features,x3 = self.features(data_dict)
        if not inference:
            pcl_map=self.PCL(x3)
            pcl_gt_map=self.Msk_PCL(self.mask_down_sampling(data_dict['mask']))
        else:
            pcl_map,pcl_gt_map = None, None
        # get the prediction by classifier
        pred,x = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]

        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'pcl_map':pcl_map, 'pcl_gt_map': pcl_gt_map}
        return pred_dict



if __name__ == '__main__':
    with open(r'H:\code\DeepfakeBench\training\config\detector\pcl_xception.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])
    detector=PCLXceptionDetector(config=config).cuda()
    config['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config['sample_size']=256
    config['with_mask']=True
    config['with_landmark']=True
    config['use_data_augmentation']=True
    train_set = I2GDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    optimizer = optim.Adam(
        params=detector.parameters(),
        lr=config['optimizer']['adam']['lr'],
        weight_decay=config['optimizer']['adam']['weight_decay'],
        betas=(config['optimizer']['adam']['beta1'], config['optimizer']['adam']['beta2']),
        eps=config['optimizer']['adam']['eps'],
        amsgrad=config['optimizer']['adam']['amsgrad'],
    )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        batch['image'],batch['label'],batch['mask']=batch['image'].cuda(),batch['label'].cuda(),batch['mask'].cuda()
        predictions=detector(batch)
        losses = detector.get_losses(batch, predictions)
        optimizer.zero_grad()
        losses['overall'].backward()
        optimizer.step()

        if iteration > 10:
            break