'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the RECCEDetector

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
@inproceedings{cao2022end,
  title={End-to-end reconstruction-classification learning for face forgery detection},
  author={Cao, Junyi and Ma, Chao and Yao, Taiping and Chen, Shen and Ding, Shouhong and Yang, Xiaokang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4113--4122},
  year={2022}
}
'''

import os
import datetime
from typing import Union
from sklearn import metrics
from collections import defaultdict
from functools import partial
from timm.models import xception
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from metrics.base_metrics_class import calculate_metrics_for_train

from networks.xception import SeparableConv2d, Block
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import logging

logger = logging.getLogger(__name__)

encoder_params = {
    "xception": {
        "features": 2048,
        "init_op": partial(xception, pretrained=True)
    }
}

@DETECTOR.register_module(module_name='recce')
class RecceDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config) # FIXME: do not use the self.backbone in recce
        self.loss_func = self.build_loss(config)
        self.model = Recce(num_classes=2)

    # FIXME: the above function should be comment or something else
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.model.features(data_dict['image'])

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.model.classifier(features)
    
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


class Recce(nn.Module):
    """ End-to-End Reconstruction-Classification Learning for Face Forgery Detection """

    def __init__(self, num_classes, drop_rate=0.2):
        super(Recce, self).__init__()
        self.name = "xception"
        self.loss_inputs = dict()
        self.encoder = encoder_params[self.name]["init_op"]()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(encoder_params[self.name]["features"], num_classes)

        self.attention = GuidedAttention(depth=728, drop_rate=drop_rate)
        self.reasoning = GraphReasoning(728, 256, 256, 256, 128, 256, [2, 4], drop_rate)

        self.decoder1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(728, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = Block(256, 256, 3, 1)
        self.decoder3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = Block(128, 128, 3, 1)
        self.decoder5 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder6 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def norm_n_corr(self, x):
        norm_embed = F.normalize(self.global_pool(x), p=2, dim=1)
        corr = (torch.matmul(norm_embed.squeeze(), norm_embed.squeeze().T) + 1.) / 2.
        return norm_embed, corr

    @staticmethod
    def add_white_noise(tensor, mean=0., std=1e-6):
        rand = torch.rand([tensor.shape[0], 1, 1, 1])
        rand = torch.where(rand > 0.5, 1., 0.).to(tensor.device)
        white_noise = torch.normal(mean, std, size=tensor.shape, device=tensor.device)
        noise_t = tensor + white_noise * rand
        noise_t = torch.clip(noise_t, -1., 1.)
        return noise_t

    def features(self, x):
                # clear the loss inputs
        self.loss_inputs = dict(recons=[], contra=[])
        noise_x = self.add_white_noise(x) if self.training else x
        out = self.encoder.conv1(noise_x)
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)
        out = self.encoder.conv2(out)
        out = self.encoder.bn2(out)
        out = self.encoder.act2(out)
        out = self.encoder.block1(out)
        out = self.encoder.block2(out)
        out = self.encoder.block3(out)
        embedding = self.encoder.block4(out)

        norm_embed, corr = self.norm_n_corr(embedding)
        self.loss_inputs['contra'].append(corr)

        out = self.dropout(embedding)
        out = self.decoder1(out)
        out_d2 = self.decoder2(out)

        norm_embed, corr = self.norm_n_corr(out_d2)
        self.loss_inputs['contra'].append(corr)

        out = self.decoder3(out_d2)
        out_d4 = self.decoder4(out)

        norm_embed, corr = self.norm_n_corr(out_d4)
        self.loss_inputs['contra'].append(corr)

        out = self.decoder5(out_d4)
        pred = self.decoder6(out)

        recons_x = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=True)
        self.loss_inputs['recons'].append(recons_x)

        embedding = self.encoder.block5(embedding)
        embedding = self.encoder.block6(embedding)
        embedding = self.encoder.block7(embedding)

        fusion = self.reasoning(embedding, out_d2, out_d4) + embedding

        embedding = self.encoder.block8(fusion)
        img_att = self.attention(x, recons_x, embedding)

        embedding = self.encoder.block9(img_att)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding)
        embedding = self.encoder.block12(embedding)

        embedding = self.encoder.conv3(embedding)
        embedding = self.encoder.bn3(embedding)
        embedding = self.encoder.act3(embedding)
        embedding = self.encoder.conv4(embedding)
        embedding = self.encoder.bn4(embedding)
        embedding = self.encoder.act4(embedding)

        embedding = self.global_pool(embedding).squeeze(2).squeeze(2)
        embedding = self.dropout(embedding)

        return embedding
    
    def classifier(self, embedding):
        return self.fc(embedding)

    def forward(self, x):
        embedding = self.features(x)
        return self.classifier(embedding)

class GraphReasoning(nn.Module):
    """ Graph Reasoning Module for information aggregation. """

    def __init__(self, va_in, va_out, vb_in, vb_out, vc_in, vc_out, spatial_ratio, drop_rate):
        super(GraphReasoning, self).__init__()
        self.ratio = spatial_ratio
        self.va_embedding = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(va_out, va_out, 1, bias=False),
        )
        self.va_gated_b = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False),
            nn.Sigmoid()
        )
        self.va_gated_c = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False),
            nn.Sigmoid()
        )
        self.vb_embedding = nn.Sequential(
            nn.Linear(vb_in, vb_out, bias=False),
            nn.ReLU(True),
            nn.Linear(vb_out, vb_out, bias=False),
        )
        self.vc_embedding = nn.Sequential(
            nn.Linear(vc_in, vc_out, bias=False),
            nn.ReLU(True),
            nn.Linear(vc_out, vc_out, bias=False),
        )
        self.unfold_b = nn.Unfold(kernel_size=spatial_ratio[0], stride=spatial_ratio[0])
        self.unfold_c = nn.Unfold(kernel_size=spatial_ratio[1], stride=spatial_ratio[1])
        self.reweight_ab = nn.Sequential(
            nn.Linear(va_out + vb_out, 1, bias=False),
            nn.ReLU(True),
            nn.Softmax(dim=1)
        )
        self.reweight_ac = nn.Sequential(
            nn.Linear(va_out + vc_out, 1, bias=False),
            nn.ReLU(True),
            nn.Softmax(dim=1)
        )
        self.reproject = nn.Sequential(
            nn.Conv2d(va_out + vb_out + vc_out, va_in, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(va_in, va_in, kernel_size=1, bias=False),
            nn.Dropout(drop_rate) if drop_rate is not None else nn.Identity(),
        )

    def forward(self, vert_a, vert_b, vert_c):
        emb_vert_a = self.va_embedding(vert_a)
        emb_vert_a = emb_vert_a.reshape([emb_vert_a.shape[0], emb_vert_a.shape[1], -1])

        gate_vert_b = 1 - self.va_gated_b(vert_a)
        gate_vert_b = gate_vert_b.reshape(*emb_vert_a.shape)
        gate_vert_c = 1 - self.va_gated_c(vert_a)
        gate_vert_c = gate_vert_c.reshape(*emb_vert_a.shape)

        vert_b = self.unfold_b(vert_b).reshape(
            [vert_b.shape[0], vert_b.shape[1], self.ratio[0] * self.ratio[0], -1])
        vert_b = vert_b.permute([0, 2, 3, 1])
        emb_vert_b = self.vb_embedding(vert_b)

        vert_c = self.unfold_c(vert_c).reshape(
            [vert_c.shape[0], vert_c.shape[1], self.ratio[1] * self.ratio[1], -1])
        vert_c = vert_c.permute([0, 2, 3, 1])
        emb_vert_c = self.vc_embedding(vert_c)

        agg_vb = list()
        agg_vc = list()
        for j in range(emb_vert_a.shape[-1]):
            # ab propagating
            emb_v_a = torch.stack([emb_vert_a[:, :, j]] * (self.ratio[0] ** 2), dim=1)
            emb_v_b = emb_vert_b[:, :, j, :]
            emb_v_ab = torch.cat([emb_v_a, emb_v_b], dim=-1)
            w = self.reweight_ab(emb_v_ab)
            agg_vb.append(torch.bmm(emb_v_b.transpose(1, 2), w).squeeze() * gate_vert_b[:, :, j])

            # ac propagating
            emb_v_a = torch.stack([emb_vert_a[:, :, j]] * (self.ratio[1] ** 2), dim=1)
            emb_v_c = emb_vert_c[:, :, j, :]
            emb_v_ac = torch.cat([emb_v_a, emb_v_c], dim=-1)
            w = self.reweight_ac(emb_v_ac)
            agg_vc.append(torch.bmm(emb_v_c.transpose(1, 2), w).squeeze() * gate_vert_c[:, :, j])

        agg_vert_b = torch.stack(agg_vb, dim=-1)
        agg_vert_c = torch.stack(agg_vc, dim=-1)
        agg_vert_bc = torch.cat([agg_vert_b, agg_vert_c], dim=1)
        agg_vert_abc = torch.cat([agg_vert_bc, emb_vert_a], dim=1)
        agg_vert_abc = torch.sigmoid(agg_vert_abc)
        agg_vert_abc = agg_vert_abc.reshape(vert_a.shape[0], -1, vert_a.shape[2], vert_a.shape[3])
        return self.reproject(agg_vert_abc)


class GuidedAttention(nn.Module):
    """ Reconstruction Guided Attention. """

    def __init__(self, depth=728, drop_rate=0.2):
        super(GuidedAttention, self).__init__()
        self.depth = depth
        self.gated = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.h = nn.Sequential(
            nn.Conv2d(depth, depth, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, pred_x, embedding):
        residual_full = torch.abs(x - pred_x)
        residual_x = F.interpolate(residual_full, size=embedding.shape[-2:],
                                   mode='bilinear', align_corners=True)
        res_map = self.gated(residual_x)
        return res_map * self.h(embedding) + self.dropout(embedding)
