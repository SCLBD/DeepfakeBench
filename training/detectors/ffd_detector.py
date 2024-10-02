'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the FFDDetector

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
@inproceedings{dang2020detection,
  title={On the detection of digital face manipulation},
  author={Dang, Hao and Liu, Feng and Stehouwer, Joel and Liu, Xiaoming and Jain, Anil K},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern recognition},
  pages={5781--5790},
  year={2020}
}

GitHub Reference:
https://github.com/JStehouwer/FFD_CVPR2020
'''

import os
import datetime
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
from imageio import imread
from torchvision import transforms
from metrics.base_metrics_class import calculate_metrics_for_train
from networks.xception import Block, SeparableConv2d
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

import logging
logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='ffd')
class FFDDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

        # model
        templates = get_templates()
        maptype = config['maptype']
        if maptype == 'none':
            self.map = [1, None]
        elif maptype == 'reg':
            self.map = RegressionMap(728)
        elif maptype == 'tmp':
            self.map = TemplateMap(728, templates)
        elif maptype == 'pca_tmp':
            self.map = PCATemplateMap(templates)
        else:
            print('Unknown map type: `{0}`'.format(maptype))

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
        cls_loss_class = LOSSFUNC[config['loss_func']['cls_loss']]
        mask_loss_class = LOSSFUNC[config['loss_func']['mask_loss']]
        cls_loss_func = cls_loss_class()
        mask_loss_func = mask_loss_class()
        loss_func = {'cls': cls_loss_func, 'mask': mask_loss_func}
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        # Pass the input through the Xception backbone
        x = self.backbone.fea_part1(data_dict['image'])
        x = self.backbone.fea_part2(x)
        x = self.backbone.fea_part3(x)  # This ends at block7 in the backbone
        mask, vec = self.map(x)  # Compute the mask here
        x = x * mask  # Apply the mask
        x = self.backbone.fea_part4(x)  # Continue with the rest of the backbone
        x = self.backbone.fea_part5(x)
        return x, mask, vec

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
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
            loss_mask = self.loss_func['mask'](pred_mask, mask_gt)
            # follow the original paper, 
            loss = loss_cls + loss_mask
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
        # get the features by backbone
        features, mask, vec = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'mask': mask, 'vec': vec}

        return pred_dict

class RegressionMap(nn.Module):
  def __init__(self, c_in):
    super(RegressionMap, self).__init__()
    self.c = SeparableConv2d(c_in, 1, 3, stride=1, padding=1, bias=False)
    self.s = nn.Sigmoid()

  def forward(self, x):
    mask = self.c(x)
    mask = self.s(mask)
    return mask, None

class TemplateMap(nn.Module):
  def __init__(self, c_in, templates):
    super(TemplateMap, self).__init__()
    self.c = Block(c_in, 364, 2, 2, start_with_relu=True, grow_first=False)
    self.l = nn.Linear(364, 10)
    self.relu = nn.ReLU(inplace=True)
    
    self.templates = templates

  def forward(self, x):
    v = self.c(x)
    v = self.relu(v)
    v = F.adaptive_avg_pool2d(v, (1,1))
    v = v.view(v.size(0), -1)
    v = self.l(v)
    mask = torch.mm(v, self.templates.reshape(10,361))
    mask = mask.reshape(x.shape[0], 1, 19, 19)

    return mask, v

class PCATemplateMap(nn.Module):
  def __init__(self, templates):
    super(PCATemplateMap, self).__init__()
    self.templates = templates

  def forward(self, x):
    fe = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
    fe = torch.transpose(fe, 1, 2)
    mu = torch.mean(fe, 2, keepdim=True)
    fea_diff = fe - mu
    
    cov_fea = torch.bmm(fea_diff, torch.transpose(fea_diff, 1, 2))
    B = self.templates.reshape(1, 10, 361).repeat(x.shape[0], 1, 1)
    D = torch.bmm(torch.bmm(B, cov_fea), torch.transpose(B, 1, 2))
    eigen_value, eigen_vector = D.symeig(eigenvectors=True)
    index = torch.tensor([9]).cuda()
    eigen = torch.index_select(eigen_vector, 2, index)

    v = eigen.squeeze(-1)
    mask = torch.mm(v, self.templates.reshape(10, 361))
    mask = mask.reshape(x.shape[0], 1, 19, 19)
    return mask, v

def get_templates():
    templates_list = []
    for i in range(10):
        img = imread('./training/lib/component/MCT/template{:d}.png'.format(i))
        templates_list.append(transforms.functional.to_tensor(img)[0:1,0:19,0:19])
    if torch.cuda.is_available():
        templates = torch.stack(templates_list).cuda()
    else:
        templates = torch.stack(templates_list)
    templates = templates.squeeze(1)
    return templates
