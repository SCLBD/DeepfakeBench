"""
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the TimesformerDetector

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
@inproceedings{bertasius2021space,
  title={Is space-time attention all you need for video understanding?},
  author={Bertasius, Gedas and Wang, Heng and Torresani, Lorenzo},
  booktitle={ICML},
  volume={2},
  number={3},
  pages={4},
  year={2021}
}
"""

import logging
import torch
import torch.nn as nn
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='timesformer')
class TimeSformerDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        # self.fc_norm = nn.LayerNorm(768)
        # self.temporal_module = self.build_temporal_module(config)
        self.head = nn.Linear(768, 2)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        from transformers import TimesformerModel
        backbone = TimesformerModel.from_pretrained(config['pretrained'])
        # for name, param in backbone.named_parameters():
        #     print('{}: {}'.format(name, param.requires_grad))
        # num_param = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        # num_total_param = sum(p.numel() for p in backbone.parameters())
        # print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
        return backbone

    def build_temporal_module(self, config):
        return nn.LSTM(input_size=2048, hidden_size=512, num_layers=3, batch_first=True)

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()

        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        # b, t, c, h, w = data_dict['image'].shape
        # frame_input = data_dict['image'].reshape(-1, c, h, w)
        # # get frame-level features
        # frame_level_features = self.backbone.features(frame_input)
        # frame_level_features = F.adaptive_avg_pool2d(frame_level_features, (1, 1)).reshape(b, t, -1)
        # # get video-level features
        # video_level_features = self.temporal_module(frame_level_features)[0][:, -1, :]

        outputs = self.backbone(data_dict['image'], output_hidden_states=True)
        video_level_features = outputs[0][:, 0]
        return video_level_features

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
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict
