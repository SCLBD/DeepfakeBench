"""
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the SIADetector

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
@inproceedings{sun2022information,
  title={An information theoretic approach for attention-driven face forgery detection},
  author={Sun, Ke and Liu, Hong and Yao, Taiping and Sun, Xiaoshuai and Chen, Shen and Ding, Shouhong and Ji, Rongrong},
  booktitle={European Conference on Computer Vision},
  pages={111--127},
  year={2022},
  organization={Springer}
}
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
from networks import BACKBONE

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='sia')
class SIADetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

        self.att0conv = SAIA_conv(24, kernel_size=3, isspace=True, ischannel=True)
        self.att1conv = SAIA_conv(32, kernel_size=3, isspace=True, ischannel=True)
        self.att2conv = SAIA_conv(56, kernel_size=3, isspace=True, ischannel=True)
        self.att3conv = SAIA_conv(112, kernel_size=3, isspace=True, ischannel=True)
        self.att4conv = SAIA_conv(160, kernel_size=3, isspace=True, ischannel=True)
        self.att5conv = SAIA_conv(272, kernel_size=3, isspace=False, ischannel=True)
        self.att6conv = SAIA_conv(448, kernel_size=3, isspace=False, ischannel=True)

        self.avgpool1 = nn.AdaptiveMaxPool2d((32, 32))
        # self.avgpool1 = nn.AdaptiveAvgPool2d((20,20))#[160]

        self.avgpool2 = nn.AdaptiveMaxPool2d((16, 16))

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 56, 1, 1, 0),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 160, 1, 1, 0),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),

        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(56, 160, 1, 1, 0),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),

        )

        num_ftrs = 1792
        num_classes = 1

        self.linear = nn.Linear(num_ftrs, num_classes)

    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        model_config['pretrained'] = self.config.get('pretrained', None)
        backbone = backbone_class(model_config)

        # FIXME: current load pretrained weights only from the backbone, not here
        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        # Extract features from the EfficientNet-B4 model
        x = data_dict['image']
        x = self.extract_features(x)
        # if self.mode == 'adjust_channel':
        #     x = self.adjust_channel(x)
        return x

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)

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

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self.backbone.efficientnet._conv_stem(inputs)
        x = self.backbone.efficientnet._bn0(x)
        x = self.backbone.efficientnet._swish(x)
        # x = self._swish(self._bn0(self._conv_stem(inputs)))

        x = self.backbone.efficientnet._blocks[0](x)
        x = self.backbone.efficientnet._blocks[1](x)
        # print("Output shape after block 1:", x.shape)

        x = self.backbone.efficientnet._blocks[2](x)
        x = self.backbone.efficientnet._blocks[3](x)
        x = self.backbone.efficientnet._blocks[4](x)
        x = self.backbone.efficientnet._blocks[5](x)
        # print("Output shape after block 5:", x.shape)

        x, att1 = self.att1conv(x)
        res12 = self.avgpool1(self.conv1(att1))
        res14 = self.avgpool2(self.conv2(att1))

        x = self.backbone.efficientnet._blocks[6](x)
        x = self.backbone.efficientnet._blocks[7](x)
        x = self.backbone.efficientnet._blocks[8](x)
        x = self.backbone.efficientnet._blocks[9](x)
        # print("Output shape after block 9:", x.shape)

        x, att2 = self.att2conv(x + res12)
        res24 = self.avgpool2(self.conv3(att2))

        x = self.backbone.efficientnet._blocks[10](x)
        x = self.backbone.efficientnet._blocks[11](x)
        x = self.backbone.efficientnet._blocks[12](x)
        x = self.backbone.efficientnet._blocks[13](x)
        x = self.backbone.efficientnet._blocks[14](x)
        x = self.backbone.efficientnet._blocks[15](x)
        # print("Output shape after block 15:", x.shape)

        x = self.backbone.efficientnet._blocks[16](x)
        x = self.backbone.efficientnet._blocks[17](x)
        x = self.backbone.efficientnet._blocks[18](x)
        x = self.backbone.efficientnet._blocks[19](x)
        x = self.backbone.efficientnet._blocks[20](x)
        x = self.backbone.efficientnet._blocks[21](x)
        # print("Output shape after block 21:", x.shape)

        x, att4 = self.att4conv(x + res24 + res14)

        x = self.backbone.efficientnet._blocks[22](x)
        x = self.backbone.efficientnet._blocks[23](x)
        x = self.backbone.efficientnet._blocks[24](x)
        x = self.backbone.efficientnet._blocks[25](x)
        x = self.backbone.efficientnet._blocks[26](x)
        x = self.backbone.efficientnet._blocks[27](x)
        x = self.backbone.efficientnet._blocks[28](x)
        x = self.backbone.efficientnet._blocks[29](x)
        # print("Output shape after block 29:", x.shape)

        x = self.backbone.efficientnet._blocks[30](x)
        x = self.backbone.efficientnet._blocks[31](x)
        # print("Output shape after block 31:", x.shape)

        # for idx, block in enumerate(self.backbone.efficientnet._blocks):
        #     drop_connect_rate = self.backbone.efficientnet._global_params.drop_connect_rate
        #     if drop_connect_rate:
        #         drop_connect_rate *= float(idx) / len(self.backbone.efficientnet._blocks)  # scale drop connect_rate
        #     x = block(x, drop_connect_rate=drop_connect_rate)
        # print(idx)

        # Head
        x = self.backbone.efficientnet._swish(self.backbone.efficientnet._bn1(self.backbone.efficientnet._conv_head(x)))

        return x


class SAIA_conv(nn.Module):
    def __init__(self, outdim, kernel_size=3, padding=1, isspace=True, ischannel=True):
        super(SAIA_conv, self).__init__()

        self.drop_rate = 0.3
        self.temperature = 0.03
        self.band_width = 1.0

        self.isspace = isspace
        self.ischannel = ischannel
        self.outdim = outdim

        kernel = torch.ones((outdim, 1, kernel_size, kernel_size))
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        kernel2 = torch.ones((outdim, 1, 1, 1)) * (kernel_size * kernel_size)
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.pad = padding
        self.channel_range = 5

    def forward(self, x):
        with torch.no_grad():
            batch_size = x.shape[0]
            num_channel = x.shape[1]
            # intra-feature

            x1 = F.conv2d(x, self.weight, padding=self.pad, groups=self.outdim)
            x2 = F.conv2d(x, self.weight2, padding=0, groups=self.outdim)
            intra_distance = torch.abs(x2 - x1)

            # inter-feature
            pad_x = torch.cat([x, x[:, :self.channel_range + 1, :, :]], dim=1)
            distances = []
            for i in range(1, self.channel_range + 1):
                tmp = (x[:, :, :, :] - pad_x[:, i:num_channel + i, :, :])
                distances.append(tmp.clone())

            distance = torch.cat(distances, dim=1)
            batch_size, _, h_dis, w_dis = distance.shape
            distance = distance.view(batch_size, -1, self.channel_range, h_dis, w_dis).sum(dim=2)
            inter_distance = torch.abs(distance.view(batch_size, -1, h_dis, w_dis))
            att = intra_distance + 0.5 * inter_distance

        if self.ischannel:
            distance_channel = att[:]
            distance_channel = torch.exp(
                -distance_channel / distance_channel.mean() / 2 / self.band_width ** 2)  # using mean of distance to normalize
            distance_channel = -torch.log(distance_channel + 0.1)
            channel_attention = torch.mean(distance_channel.view(batch_size, self.outdim, -1), dim=2)
            channel_attention = channel_attention.view(batch_size, -1, 1, 1) + 1

        if self.isspace:
            distance_space = att
            distance_space = distance_space / distance_space.mean() / 2 / self.band_width ** 2
            space_attention = distance_space
            batch_size, channels, h, w = x.shape
            attention_image = (nn.Sigmoid()(space_attention) + 1) * x

        if self.isspace and self.ischannel:
            return attention_image * (channel_attention.expand_as(x)), space_attention
        elif self.isspace:
            return attention_image, x
        elif self.ischannel:
            return x * (channel_attention.expand_as(x)), x
