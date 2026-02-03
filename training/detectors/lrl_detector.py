'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the LocalRelationDetector

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
@inproceedings{chen2021local,
  title={Local relation learning for face forgery detection},
  author={Chen, Shen and Yao, Taiping and Chen, Yang and Ding, Shouhong and Li, Jilin and Ji, Rongrong},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={35},
  number={2},
  pages={1081--1088},
  year={2021}
}
'''

import os
import datetime
import logging
import numpy as np
import yaml
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel, Dropout2d, UpsamplingBilinear2d
from torch.utils.tensorboard import SummaryWriter

from dataset.lrl_dataset import LRLDataset
from metrics.base_metrics_class import calculate_metrics_for_train

from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import random

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='lrl')
class LRLDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_rgb = self.build_backbone(config)
        self.encoder_idct = self.build_backbone(config)
        self.encoder_idct.efficientnet._conv_stem = nn.Conv2d(1, 48, kernel_size=3, stride=2, bias=False)
        self.loss_func = self.build_loss(config)
        self.feature_adjust1 = nn.Upsample(scale_factor=0.25)
        self.feature_adjust2 = nn.Upsample(scale_factor=0.5)
        self.decoder = Decoder(decoder_filters=[64, 128, 256, 256],
                                    filters=[48, 40, 64, 176, 2008])
        self.rfam1 = RFAM(56)
        self.rfam2 = RFAM(160)
        self.rfam3 = RFAM(1792)

        self.final = nn.Conv2d(64, out_channels=1, kernel_size=1, bias=False)

        self.overall_classifier = nn.Sequential(
            nn.Linear(240, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        model_config['pretrained'] = self.config['pretrained']
        backbone = backbone_class(model_config)
        if config['pretrained'] != 'None':
            logger.info('Load pretrained model successfully!')
        else:
            logger.info('No pretrained model.')
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        self.seg_loss = nn.BCELoss()
        self.sim_loss = nn.MSELoss()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        rgb=data_dict['image']
        idct=data_dict['idct']

        #torch.Size([b, 56, 32, 32])
        rgb1=self.encoder_rgb.block_part1(rgb)
        idct1=self.encoder_idct.block_part1(idct)

        rgb1, idct1 = self.rfam1(rgb1, idct1)
        featuremap_low = rgb1 + idct1

        #torch.Size([b, 160, 16, 16])
        rgb2=self.encoder_rgb.block_part2(rgb1)
        idct2=self.encoder_idct.block_part2(idct1)

        rgb2, idct2 = self.rfam2(rgb2, idct2)
        featuremap_mid = rgb2 + idct2

        #torch.Size([b, 1792, 8, 8])
        rgb3=self.encoder_rgb.block_part3(rgb2)
        idct3=self.encoder_idct.block_part3(idct2)

        rgb3, idct3 = self.rfam3(rgb3, idct3)
        featuremap_high = rgb3 + idct3

        f1 = self.feature_adjust1(featuremap_low)
        f2 = self.feature_adjust2(featuremap_mid)
        f3 = featuremap_high
        featuremap = torch.cat((f1, f2, f3), dim=1)

        return featuremap

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.overall_classifier(features)

    def get_similaritys(self, masks, n=9, m=9):
        similaritys = []
        for i in range(len(masks)):
            ratios = [y.float().mean() for x in torch.chunk(masks[i], n, dim=0) for y in torch.chunk(x, m, dim=1)]
            ratios = torch.tensor(ratios).view(-1, 1)
            similarity = 1 - torch.norm(ratios[:, None] - ratios, dim=2, p=2)
            similaritys.append(similarity)
        similaritys = torch.stack(similaritys)
        return similaritys

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        masks = data_dict['mask']
        pred_mask = pred_dict['mask_pred']
        sim = pred_dict['sim']
        sim_gt = self.get_similaritys(masks.squeeze(1), n=4, m=4).cuda()
        pred = pred_dict['cls']
        sim_loss = self.sim_loss(sim,sim_gt)
        seg_loss = self.seg_loss(pred_mask,masks)
        ce_loss = self.loss_func(pred, label)
        loss = sim_loss+seg_loss+ce_loss
        loss_dict = {'overall': loss,'sim_loss':sim_loss,'seg_loss':seg_loss,'ce_loss':ce_loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = torch.ceil(data_dict['label'].clamp(max=1).float()).long()
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def feature_process(self,feature):
        w = F.unfold(feature, kernel_size=2, stride=2, padding=0).permute(0, 2, 1) # (2008,8,8) to (16,8032), that is, 4*4 and flatten
        w_normed = w / (w * w).sum(dim=2, keepdim=True).sqrt()
        B, K = w.shape[:2]
        sim = torch.einsum('bij,bjk->bik', w_normed, w_normed.permute(0, 2, 1)) # cross-similarity (16,16)
        sim = (sim + 1) / 2
        mask = (torch.eye(K) != 1).repeat(B, 1).view(B, K, K).cuda()
        sim_mask = torch.masked_select(sim, mask).view(B, K, -1) # remove self-similarity
        x = sim_mask.view(B, -1)
        return x,sim

    def forward(self, data_dict: dict, inference=False) -> dict:

        # get the features by backbone
        features = self.features(data_dict)

        features_processed,sim = self.feature_process(features)
        # get the prediction by classifier
        pred_raw = self.classifier(features_processed)

        encoder_results = [features]
        mask = self.final(self.decoder(encoder_results))
        mask = torch.sigmoid(mask)
        # get the probability of the pred
        if pred_raw.size(1)>2:
            pred=torch.stack([pred_raw[:, 0], torch.sum(pred_raw[:, 1:], dim=1)], dim=1)
        else:
            pred=pred_raw
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred_raw, 'prob': prob, 'feat': features, 'mask_pred': mask, 'sim': sim}
        return pred_dict
        # else:
        #     return pred
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class ConcatBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc=None):
        return self.seq(dec)


class Decoder(nn.Module):
    def __init__(self, decoder_filters, filters, upsample_filters=None,
                 decoder_block=DecoderBlock, bottleneck=ConcatBottleneck, dropout=0):
        super().__init__()
        self.decoder_filters = decoder_filters
        self.filters = filters
        self.decoder_block = decoder_block
        self.decoder_stages = nn.ModuleList([self._get_decoder(idx) for idx in range(0, len(decoder_filters))])
        self.bottlenecks = nn.ModuleList([bottleneck(f, f)
                                          for i, f in enumerate(reversed(decoder_filters))])
        self.dropout = Dropout2d(dropout) if dropout > 0 else None
        self.last_block = None
        if upsample_filters:
            self.last_block = decoder_block(decoder_filters[0], out_channels=upsample_filters)
        else:
            self.last_block = UpsamplingBilinear2d(scale_factor=2)

    def forward(self, encoder_results: list):
        x = encoder_results[0]
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x)
        if self.last_block:
            x = self.last_block(x)
        if self.dropout:
            x = self.dropout(x)
        return x

    def _get_decoder(self, layer):
        idx = layer + 1
        if idx == len(self.decoder_filters):
            in_channels = self.filters[idx]
        else:
            in_channels = self.decoder_filters[idx]
        return self.decoder_block(in_channels, self.decoder_filters[max(layer, 0)])


class RFAM(nn.Module):
    def __init__(self, features):
        super(RFAM, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(features * 2, features, 1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, 2, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        U = torch.cat((x1, x2), dim=1)
        A = self.attention(U)
        A1 = A[:, 0, ...].unsqueeze(1).contiguous()
        A2 = A[:, 1, ...].unsqueeze(1).contiguous()
        x1 *= A1
        x2 *= A2
        return x1, x2


if __name__ == '__main__':

    with open(r'H:\code\DeepfakeBench\training\config\detector\lrl.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])
    detector=LRLDetector(config=config).cuda()
    config['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config['sample_size']=256
    config['with_mask']=True
    config['with_landmark']=True
    config['use_data_augmentation']=True
    train_set = LRLDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    optimizer = optim.Adam(
        params=detector.parameters(),
        lr=config['optimizer']['adam']['lr'],
        weight_decay=config['optimizer']['adam']['weight_decay'],
        betas=(config['optimizer']['adam']['beta1'], config['optimizer']['adam']['beta2']),
        eps=config['optimizer']['adam']['eps'],
        amsgrad=config['optimizer']['adam']['amsgrad'],
    )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        batch['image'],batch['label'],batch['mask'],batch['idct']=batch['image'].cuda(),batch['label'].cuda(),batch['mask'].cuda(),batch['idct'].cuda()
        predictions=detector(batch)
        losses = detector.get_losses(batch, predictions)
        optimizer.zero_grad()
        losses['overall'].backward()
        optimizer.step()

        if iteration > 10:
            break
