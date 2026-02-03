'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the CoreDetector

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
@inproceedings{ni2022core,
  title={Core: Consistent representation learning for face forgery detection},
  author={Ni, Yunsheng and Meng, Depu and Yu, Changqian and Quan, Chengbin and Ren, Dongchun and Zhao, Youjian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12--21},
  year={2022}
}

GitHub Reference:
https://github.com/nii-yamagishilab/Capsule-Forensics-v2
'''

import os
import datetime
import logging
import random
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
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='core')
class CoreDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model successfully!')
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image'])

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        core_feat = pred_dict['core_feat']
        loss = self.loss_func(core_feat, pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the core_feat for loss
        core_feat = nn.ReLU(inplace=False)(features)
        core_feat= F.adaptive_avg_pool2d(core_feat, (1, 1))
        core_feat = core_feat.view(core_feat.size(0), -1)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'core_feat': core_feat}

        return pred_dict

