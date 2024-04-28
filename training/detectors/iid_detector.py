'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the IIDDetector

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
@inproceedings{huang2023implicit,
  title={Implicit identity driven deepfake face swapping detection},
  author={Huang, Baojin and Wang, Zhongyuan and Yang, Jifan and Ai, Jiaxin and Zou, Qin and Wang, Qian and Ye, Dengpan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4490--4499},
  year={2023}
}
'''

import os
import datetime
import logging
import random

import numpy as np
import yaml
from sklearn import metrics
from typing import Union
from collections import defaultdict

from dataset.iid_dataset import IIDDataset
from detectors.utils.iid_api import FC_ddp,FC_ddp2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train
from networks.iresnet_iid import iresnet50

from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from .utils.iid_api import l2_norm

logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)


@DETECTOR.register_module(module_name='iid')
class IIDDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.explicit_extractor = iresnet50(False, fp16=False)
        self.explicit_extractor.load_state_dict(torch.load(config['explicit_extractor_pretrained']))
        self.explicit_extractor.cuda().eval()
        self.BCE_LOSS = FC_ddp(config['embedding_size'], config['backbone_config']['num_classes']).cuda()
        self.IIE_LOSS = FC_ddp2(config['embedding_size'], 1000, scale=64, margin=0.4, mode='arcface', use_cifp=False,
                       reduction='mean',ddp=config['ddp']).cuda()
        self.IIE_LOSS.train().cuda()
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        if config['pretrained'] != 'None':
            # if donot load the pretrained weights, fail to get good results
            state_dict = torch.load(config['pretrained'])
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
            backbone.load_state_dict(state_dict, False)
            logger.info('Load pretrained model successfully!')
        else:
            logger.info('No pretrained model.')
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        if config['loss_func']=='center_loss':
            loss_func = loss_class(num_classes=2, feat_dim=2048)
        else:
            loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor,id_f=None) -> torch.tensor:
        return self.backbone.classifier(features,id_f)

    def get_train_loss(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        id_index = data_dict['id_index'].cuda()
        id_feat = pred_dict['id_feat']
        embed = pred_dict['embed']

        real_id = (label == 1)
        fake_id = (label == 0)
        im_embs = l2_norm(embed)
        em_embs = l2_norm(id_feat)
        loss = 0

        loss_eic = (im_embs[real_id] * em_embs[real_id]).sum(dim=1).mean() - (im_embs[fake_id] * em_embs[fake_id]).sum(
            dim=1).mean()
        loss_ce = self.BCE_LOSS(pred, label, return_logits=True).mean()
        loss += loss_ce
        loss_id, _, _ = self.IIE_LOSS(embed, id_index, return_logits=True)
        # loss_id = 0
        loss += 0.05 * loss_id
        loss += 0.1 * loss_eic

        loss_dict = {'overall': loss,'loss_bce': loss_ce, 'loss_iie': loss_id, 'loss_eic': loss_eic}
        return loss_dict

    def get_test_loss(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        id_feat = pred_dict['id_feat']
        embed = pred_dict['embed']

        real_id = (label == 1)
        fake_id = (label == 0)
        im_embs = l2_norm(embed)
        em_embs = l2_norm(id_feat)
        loss = 0

        loss_eic = (im_embs[real_id] * em_embs[real_id]).sum(dim=1).mean() - (im_embs[fake_id] * em_embs[fake_id]).sum(
            dim=1).mean()
        loss_ce = self.BCE_LOSS(pred, label, return_logits=True).mean()
        loss += loss_ce
        # loss_id = 0
        loss += 0.1 * loss_eic

        loss_dict = {'overall': loss,'loss_bce': loss_ce, 'loss_iie': 0, 'loss_eic': loss_eic}
        return loss_dict

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        if 'id_index' in data_dict: # depend on the dataset for io
            return self.get_train_loss(data_dict,pred_dict)
        else:
            return self.get_test_loss(data_dict, pred_dict)
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        resized_images = F.interpolate(data_dict['image'], size=(112, 112), mode='bilinear', align_corners=False)
        id_feat = self.explicit_extractor(resized_images)
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features,id_feat)

        embed=self.backbone.last_emb
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]

        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features,'id_feat': id_feat,'embed':embed}
        return pred_dict

if __name__ == '__main__':

    with open(r'H:\code\DeepfakeBench\training\config\detector\iid_detector.yaml', 'r') as f:
        config = yaml.safe_load(f)
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])

    config['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config['sample_size']=256
    config['with_mask']=False
    config['with_landmark']=False
    config['use_data_augmentation']=True
    config['ddp']=False
    detector=IIDDetector(config=config).cuda()
    train_set = IIDDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=4,
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
        batch['image'],batch['label']=batch['image'].cuda(),batch['label'].cuda()
        predictions=detector(batch)
        losses = detector.get_losses(batch, predictions)
        optimizer.zero_grad()
        losses['overall'].backward()
        optimizer.step()

        if iteration > 10:
            break
