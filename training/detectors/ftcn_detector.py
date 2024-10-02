config_text = """
TRAIN:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
  EVAL_PERIOD: 10
  CHECKPOINT_PERIOD: 1
  AUTO_RESUME: True
DATA:
  NUM_FRAMES: 16
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
# description: Class for the XceptionDetector

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
@inproceedings{rossler2019faceforensics++,
  title={Faceforensics++: Learning to detect manipulated facial images},
  author={Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1--11},
  year={2019}
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
from inspect import signature
from networks.time_transformer import TimeTransformer
import random


random_select = True
no_time_pool = True

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='ftcn')
class FTCNDetector(AbstractDetector):
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
        

        temporal_only_conv(self.resnet, "model", 0)

        stop_point = 5
        for i in [5, 4, 3]:
            if stop_point <= i:
                setattr(self.resnet, f"s{i}", nn.Identity())
                if stop_point==3:
                    setattr(self.resnet, f"pathway0_pool", nn.Identity())
        
        params = {
            6: dict(spatial_size=7, time_size=config['clip_size'], in_channels=2048),
            5: dict(spatial_size=14, time_size=config['clip_size'], in_channels=1024),
            4: dict(spatial_size=28, time_size=config['clip_size'], in_channels=512),
            3: dict(spatial_size=56, time_size=config['clip_size']*2, in_channels=256),
        }[stop_point]

        self.resnet.head = TransformerHead(**params)

        self.loss_func = nn.BCELoss()  # The output of the model is a probability value between 0 and 1 (haved used sigmoid)
    
    def build_backbone(self, config):
        pass
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        inputs = [data_dict['image'].permute(0,2,1,3,4)]
        pred, video_level_features = self.resnet(inputs)
        output = {}
        output["final_output"] = pred
        return output["final_output"], video_level_features

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
        # get the features and probability
        prob, features = self.features(data_dict)
        # build the prediction dict for each output
        pred_dict = {'cls': prob, 'prob': prob, 'feat': features}
        return pred_dict




class RandomPatchPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        if self.training and random_select:
            while True:
                idx = random.randint(0, h * w - 1)
                i = idx // h
                j = idx % h
                if j == 0 or i == h - 1 or j == h - 1:
                    continue
                else:
                    break
        else:
            idx = h * w // 2
        x = x[..., idx]
        return x


def valid_idx(idx, h):
    i = idx // h
    j = idx % h
    if j == 0 or i == h - 1 or j == h - 1:
        return False
    else:
        return True


class RandomAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        candidates = list(range(h * w))
        candidates = [idx for idx in candidates if valid_idx(idx, h)]
        max_k = len(candidates)
        if self.training and random_select:
            k = 8
        else:
            k = max_k
        candidates = random.sample(candidates, k)
        x = x[..., candidates].mean(-1)
        return x


class TransformerHead(nn.Module):
    def __init__(self, spatial_size=7, time_size=16, in_channels=2048):
        super().__init__()
        # if no_time_pool:
        #     time_size = time_size * 2
        patch_type = "time"
        if patch_type == "time":
            self.pool = nn.AvgPool3d((1, spatial_size, spatial_size))
            self.num_patches = time_size
        elif patch_type == "spatial":
            self.pool = nn.AvgPool3d((time_size, 1, 1))
            self.num_patches = spatial_size ** 2
        elif patch_type == "random":
            self.pool = RandomPatchPool()
            self.num_patches = time_size
        elif patch_type == "random_avg":
            self.pool = RandomAvgPool()
            self.num_patches = time_size
        elif patch_type == "all":
            self.pool = nn.Identity()
            self.num_patches = time_size * spatial_size * spatial_size
        else:
            raise NotImplementedError(patch_type)

        self.dim = -1
        if self.dim == -1:
            self.dim = in_channels

        self.in_channels = in_channels

        if self.dim != self.in_channels:
            self.fc = nn.Linear(self.in_channels, self.dim)

        default_params = dict(
            dim=self.dim, depth=1, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
        )
        params = dict(
            patch_type="time", 
            stop_point=5,
            random_select=True,
            k=8,
            sigmoid_before=False,
        )
        for key in default_params:
            if key in params:
                default_params[key] = params[key]
        print(default_params)
        self.time_T = TimeTransformer(
            num_patches=self.num_patches, num_classes=1, **default_params
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.pool(x[0])
        x = feat.reshape(-1, self.in_channels, self.num_patches)
        x = x.permute(0, 2, 1)
        if self.dim != self.in_channels:
            x = self.fc(x.reshape(-1, self.in_channels))
            x = x.reshape(-1, self.num_patches, self.dim)
        x = self.time_T(x)
        x = self.sigmoid(x)
        return x, feat


parameters = [parameter for parameter in signature(nn.Conv3d).parameters]
print(parameters)

spatial_count = 0
keep_stride_count = 0
print(f"spatial_count={spatial_count} keep_stride_count={keep_stride_count}")


def temporal_only_conv(module, name, removed, stride_removed=0):
    """
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    """
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        sub_module = getattr(module, attr_str)
        if type(sub_module) == nn.Conv3d:
            target_spatial_size = 1
            predefine_padding = {1: 0, 3: 1, 5: 2, 7: 3}
            kernel_size = list(sub_module.kernel_size)
            assert kernel_size[1] == kernel_size[2]
            stride = sub_module.stride
            extra = None
            if stride[1] == stride[2] == 2:
                stride_removed += 1
                if stride_removed > keep_stride_count:
                    stride = [1, 1, 1]
                    extra = nn.MaxPool3d((1, 2, 2))
                else:
                    print(f"stride {stride_removed} keeped")

            if kernel_size[1] == 1 and extra is None:
                continue
            padding = list(sub_module.padding)

            kernel_size[1] = kernel_size[2] = target_spatial_size
            padding[1] = padding[2] = predefine_padding[target_spatial_size]

            # param_dict = {key: getattr(sub_module, key) for key in parameters }
            param_dict = {key: getattr(sub_module, key) for key in parameters if key not in ['device', 'dtype']}

            param_dict.update(kernel_size=kernel_size, padding=padding, stride=stride)

            conv = nn.Conv3d(**param_dict)

            new_module = conv

            removed += 1
            if removed > spatial_count:
                print(
                    f"{removed} replace {name}.{attr_str}: {str(sub_module)} with {str(new_module)}"
                )
                setattr(module, attr_str, new_module)
                if extra is not None:
                    if attr_str == "conv":
                        bn_str = "bn"
                    else:
                        bn_str = f"{attr_str}_bn"
                    if hasattr(module, bn_str):
                        bn_module = getattr(module, bn_str)
                        assert isinstance(bn_module, nn.BatchNorm3d)
                        new_bn_module = nn.Sequential(bn_module, extra)
                        setattr(module, bn_str, new_bn_module)
                        print(f"stride {stride_removed} replace {name}.{bn_str}: {str(new_bn_module)}")
                    else:
                        print(f"Attribute {bn_str} not found in {name}")
            else:
                print("keep spatial")
        elif type(sub_module) == nn.Dropout:
            new_module = nn.Dropout(p=0.5)
            # print(f"replace {name}.{attr_str}: {str(sub_module)} with {str(new_module)}")
            setattr(module, attr_str, new_module)
        if no_time_pool:
            if type(sub_module) == nn.MaxPool3d:
                kernel_size = list(sub_module.kernel_size)
                if kernel_size[0] == 2:
                    kernel_size[0] = 1
                    setattr(module, attr_str, nn.MaxPool3d(kernel_size))
            elif type(sub_module) == nn.AvgPool3d:
                kernel_size = list(sub_module.kernel_size)
                kernel_size[0] = 2 * kernel_size[0]
                setattr(module, attr_str, nn.AvgPool3d(kernel_size))

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    old_name = name
    for name, immediate_child_module in module.named_children():
        removed, stride_removed = temporal_only_conv(
            immediate_child_module, old_name + "." + name, removed, stride_removed
        )
    return removed, stride_removed


# class I3D8x8(nn.Module):
#     def __init__(self, pretrained_path=None) -> None:
#         super(I3D8x8, self).__init__()
#         cfg = get_cfg()
#         cfg.merge_from_str(config_text)
#         cfg.NUM_GPUS = 1
#         cfg.TEST.BATCH_SIZE = 1
#         cfg.TRAIN.BATCH_SIZE = 1
#         cfg.DATA.NUM_FRAMES = 16
#         self.resnet = ResNetOri(cfg)
#         temporal_only_conv(self.resnet, "model", 0)

#         stop_point = 5

#         for i in [5, 4, 3]:
#             if stop_point <= i:
#                 setattr(self.resnet, f"s{i}", nn.Identity())
#                 if stop_point==3:
#                     setattr(self.resnet, f"pathway0_pool", nn.Identity())
        
#         params = {
#             6: dict(spatial_size=7, time_size=16, in_channels=2048),
#             5: dict(spatial_size=14, time_size=16, in_channels=1024),
#             4: dict(spatial_size=28, time_size=16, in_channels=512),
#             3: dict(spatial_size=56, time_size=32, in_channels=256),
#         }[stop_point]

#         self.resnet.head = TransformerHead(**params)

#         if pretrained_path is not None:
#             print(f"loading pretrained model from {pretrained_path}")
#             pretrained_weights = torch.load(pretrained_path)
#             modified_weights = {k.replace("resnet.", ""): v for k, v in pretrained_weights.items()}
#             self.resnet.load_state_dict(modified_weights, strict=True)

#     def forward(
#         self,
#         images,
#         noise=None,
#         has_mask=None,
#         freeze_backbone=False,
#         return_feature_maps=False,
#     ):
#         assert not freeze_backbone
#         inputs = [images]
#         pred = self.resnet(inputs)
#         output = {}
#         output["final_output"] = pred
#         return output


# if __name__ == '__main__':
#     model = I3D8x8()
#     inp = torch.randn(1, 3, 16, 224, 224)
#     out = model(inp)
