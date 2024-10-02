'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the UCFDetector

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
@inproceedings{li2020face,
  title={Face x-ray for more general face forgery detection},
  author={Li, Lingzhi and Bao, Jianmin and Zhang, Ting and Yang, Hao and Chen, Dong and Wen, Fang and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5001--5010},
  year={2020}
}

Notes:
To implement Face X-ray, we utilize the pretrained hrnetv2_w48 as the backbone. Despite our efforts to experiment with alternative backbones, we were unable to attain comparable results with other detectors.
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from networks.cls_hrnet import get_cls_net
import yaml

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='facexray')
class FaceXrayDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # build model
        self.backbone = self.build_backbone(config)
        self.post_process = nn.Sequential(
            nn.Conv2d(in_channels=720, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self.loss_func = self.build_loss(config)
    
    def build_backbone(self, config):
        cfg_path = './training/config/backbone/cls_hrnet_w48.yaml'
        # parse options and load config
        with open(cfg_path, 'r') as f:
            cfg_config = yaml.safe_load(f)
        convnet = get_cls_net(cfg_config)
        saved = torch.load('./training/pretrained/hrnetv2_w48_imagenet_pretrained.pth', map_location='cpu')
        convnet.load_state_dict(saved, False)
        print('Load HRnet')
        return convnet
    
    def build_loss(self, config):
        cls_loss_class = LOSSFUNC[config['loss_func']['cls_loss']]
        mask_loss_class = LOSSFUNC[config['loss_func']['mask_loss']]
        cls_loss_func = cls_loss_class()
        mask_loss_func = mask_loss_class()
        loss_func = {'cls': cls_loss_func, 'mask': mask_loss_func}
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image'])

    def classifier(self, features: list) -> torch.tensor:
        # mask
        mask = self.post_process(features)
        # feat
        feat = F.adaptive_avg_pool2d(mask, 128).view(mask.size(0), -1)
        # cls
        score = self.fc(feat)
        return feat, score, mask

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # label
        label = data_dict['label']
        mask_gt = data_dict['mask'] if data_dict['mask'] is not None else None
        # pred
        pred_cls = pred_dict['cls']
        pred_mask = pred_dict['mask_pred'] if data_dict['mask'] is not None else None
        # loss
        loss_cls = self.loss_func['cls'](pred_cls, label)
        if data_dict['mask'] is not None:
            # Move tensors to the same device
            mask_gt = mask_gt.to(pred_mask.device)
            loss_mask = F.mse_loss(pred_mask.squeeze().float(), mask_gt.squeeze().float())
            # follow the original paper, 
            # FIXME: we set λ = 1000 to force the network focusing more on learning the face X-ray prediction
            loss = loss_cls + 1000. * loss_mask
            loss_dict = {'overall': loss, 'mask': loss_mask, 'cls': loss_cls}
        else:  # mask_gt is none (during the testing or inference)
            loss = loss_cls
            loss_dict = {'overall': loss, 'cls': loss_cls}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        features, pred, mask_pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'mask_pred': mask_pred}

        return pred_dict

