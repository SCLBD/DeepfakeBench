'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the CLIPDetector

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
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
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
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig
import loralib as lora
import copy

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='clip')
class CLIPDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(768, 2)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # prepare the backbone
        _, backbone = get_clip_visual(model_name="openai/clip-vit-base-patch16")
        return backbone

        
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
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
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict


def get_clip_visual(model_name = "openai/clip-vit-base-patch16"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model
