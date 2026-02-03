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
# description: Class for the I3DDetector

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
@inproceedings{carreira2017quo,
  title={Quo vadis, action recognition? a new model and the kinetics dataset},
  author={Carreira, Joao and Zisserman, Andrew},
  booktitle={proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6299--6308},
  year={2017}
}
'''

import logging
import os
import sys

from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import torch
from .utils.slowfast.models.video_model_builder import ResNet as ResNetOri
from .utils.slowfast.config.defaults import get_cfg
from torch import nn

random_select = True
no_time_pool = True

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='i3d')
class I3DDetector(AbstractDetector):
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

        self.loss_func = nn.BCELoss()  # The output of the model is a probability value between 0 and 1 (haved used sigmoid)

    def build_backbone(self, config):
        pass

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        inputs = [data_dict['image'].permute(0, 2, 1, 3, 4)]
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
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        prob = self.features(data_dict)
        pred_dict = {'cls': prob, 'prob': prob, 'feat': prob}

        return pred_dict
