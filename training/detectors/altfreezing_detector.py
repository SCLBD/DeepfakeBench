config_text = """
TRAIN:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
  EVAL_PERIOD: 10
  CHECKPOINT_PERIOD: 1
  AUTO_RESUME: True
DATA:
  NUM_FRAMES: 8
  SAMPLING_RATE: 8
  TRAIN_JITTER_SCALES: [256, 320]
  TRAIN_CROP_SIZE: 224
  TEST_CROP_SIZE: 256
  INPUT_CHANNEL_NUM: [3]
RESNET:
  ZERO_INIT_FINAL_BN: True
  WIDTH_PER_GROUP: 64
  NUM_GROUPS: 1
  DEPTH: 50
  TRANS_FUNC: bottleneck_transform
  STRIDE_1X1: False
  NUM_BLOCK_TEMP_KERNEL: [[3], [4], [6], [3]]
NONLOCAL:
  LOCATION: [[[]], [[]], [[]], [[]]]
  GROUP: [[1], [1], [1], [1]]
  INSTANTIATION: softmax
BN:
  USE_PRECISE_STATS: True
  NUM_BATCHES_PRECISE: 200
SOLVER:
  BASE_LR: 0.1
  LR_POLICY: cosine
  MAX_EPOCH: 196
  MOMENTUM: 0.9
  WEIGHT_DECAY: 1e-4
  WARMUP_EPOCHS: 34.0
  WARMUP_START_LR: 0.01
  OPTIMIZING_METHOD: sgd
MODEL:
  NUM_CLASSES: 1
  ARCH: i3d
  MODEL_NAME: ResNet
  LOSS_FUNC: cross_entropy
  DROPOUT_RATE: 0.5
  HEAD_ACT: sigmoid
TEST:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
DATA_LOADER:
  NUM_WORKERS: 8
  PIN_MEMORY: True
NUM_GPUS: 8
NUM_SHARDS: 1
RNG_SEED: 0
OUTPUT_DIR: .
"""


'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the AltFreezingDetector

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
@InProceedings{Wang_2023_CVPR,
    author    = {Wang, Zhendong and Bao, Jianmin and Zhou, Wengang and Wang, Weilun and Li, Houqiang},
    title     = {AltFreezing for More General Video Face Forgery Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {4129-4138}
}
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


import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import torch
from .utils.slowfast.models.video_model_builder import ResNet as ResNetOri
from .utils.slowfast.config.defaults import get_cfg
from torch import nn
import random


random_select = True
no_time_pool = False

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='altfreezing')
class AltFreezingDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        cfg = get_cfg()
        cfg.merge_from_str(config_text)
        cfg.NUM_GPUS = 1
        cfg.TEST.BATCH_SIZE = 1
        cfg.TRAIN.BATCH_SIZE = 1
        cfg.DATA.NUM_FRAMES = config['clip_size']
        self.resnet = ResNetOri(cfg)
        if config['pretrained'] is not None:
            print(f"loading pretrained model from {config['pretrained']}")
            pretrained_weights = torch.load(config['pretrained'], map_location='cpu', encoding='latin1')
            modified_weights = {k.replace("resnet.", ""): v for k, v in pretrained_weights.items()}
            # fit from 400 num_classes to 1
            modified_weights["head.projection.weight"] = modified_weights["head.projection.weight"][:1, :]
            modified_weights["head.projection.bias"] = modified_weights["head.projection.bias"][:1]
            # load final ckpt
            self.resnet.load_state_dict(modified_weights, strict=True)

        self.conv_dict = self.find_conv_layers(self.resnet)
        print("1x3x3 Conv: {}\n3x1x1 Conv:{}".format(self.conv_dict['spatial'], self.conv_dict['temporal']))
        self.train_batch_cnt = 0

        self.loss_func = nn.BCELoss()  # The output of the model is a probability value between 0 and 1 (haved used sigmoid)

    def find_conv_layers(self, module, parent_name='', conv_dict=None):
        if conv_dict is None:
            conv_dict = {'temporal': [], 'spatial': []}

        for name, sub_module in module.named_children():
            full_name = f'{parent_name}.{name}' if parent_name else name

            if isinstance(sub_module, nn.Conv3d):
                if sub_module.kernel_size == (3, 1, 1):
                    conv_dict['temporal'].append(full_name)
                if sub_module.kernel_size == (1, 3, 3):
                    conv_dict['spatial'].append(full_name)
            else:
                self.find_conv_layers(sub_module, full_name, conv_dict)

        return conv_dict

    def alternate_mode(self, target_mode):
        for layer_name in self.conv_dict['temporal']:
            layer = dict(self.resnet.named_modules())[layer_name]
            layer.weight.requires_grad = True if target_mode == 'temporal' else False
            if layer.bias is not None:
                layer.bias.requires_grad = True if target_mode == 'temporal' else False

        for layer_name in self.conv_dict['spatial']:
            layer = dict(self.resnet.named_modules())[layer_name]
            layer.weight.requires_grad = True if target_mode == 'spatial' else False
            if layer.bias is not None:
                layer.bias.requires_grad = True if target_mode == 'spatial' else False


    def build_backbone(self, config):
        pass
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()

        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        inputs = [data_dict['image'].permute(0,2,1,3,4)]
        pred = self.resnet(inputs)
        output = {"final_output": pred}

        return output["final_output"]

    def classifier(self, features: torch.tensor):
        pass
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'].float()
        pred = pred_dict['cls'].view(-1)
        loss = self.loss_func(pred, label)
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
        # get the probability
        prob = self.features(data_dict)
        # build the prediction dict for each output
        pred_dict = {'cls': prob, 'prob': prob, 'feat': prob}
        if not inference:
            if self.train_batch_cnt % (20 + 1) == 0:
                self.alternate_mode('spatial')
            elif self.train_batch_cnt % (20 + 1) == 1:
                self.alternate_mode('temporal')

            self.train_batch_cnt += 1
        
        return pred_dict
