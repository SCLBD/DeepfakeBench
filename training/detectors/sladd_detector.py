'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the SLADDDetector

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
@inproceedings{chen2022self,
  title={Self-supervised learning of adversarial example: Towards good generalizations for deepfake detection},
  author={Chen, Liang and Zhang, Yong and Song, Yibing and Liu, Lingqiao and Wang, Jue},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={18710--18719},
  year={2022}
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from dataset.pair_dataset import pairDataset
from metrics.base_metrics_class import calculate_metrics_for_train


from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from .utils.sladd_api import synthesizer

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

@DETECTOR.register_module(module_name='sladd')
class SLADDXceptionDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.synthesizer = synthesizer(config=config)
        params_synthesizer = ([p for p in self.synthesizer.parameters()])
        # train
        self.optimizer_synthesizer = optim.Adam(params_synthesizer, lr=config['optimizer']['adam']['lr']/4, betas=(config['optimizer']['adam']['beta1']/4, 0.999),
                                        weight_decay=config['optimizer']['adam']['weight_decay'])

    # synthesizer should be optimized solely ---> according to the official code.
    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'synthesizer' not in name:
                yield param

    def get_test_metrics(self):
        pass

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
        self.l1loss = nn.MSELoss()
        self.cls_criterion = LOSSFUNC[config['typeloss_func']]()
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, imgs) -> torch.tensor:
        return self.backbone.features(imgs) #32,3,256,256

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = pred_dict['label']
        pred = pred_dict['cls']
        if 'map' in pred_dict:
            map, type, mag, type_label, mag_mask, mag_label, alt_mask\
                = pred_dict['map'],pred_dict['type'],pred_dict['mag'],pred_dict['type_label'],\
                pred_dict['mag_mask'],pred_dict['mag_label'],pred_dict['alt_mask']
            loss_type = self.cls_criterion(type, type_label).mean()
            loss_mag = self.l1loss(mag*mag_mask, mag_label*mag_mask).mean()
            loss_maps = self.l1loss(map, alt_mask)

        else:
            loss_type,loss_mag,loss_maps=0,0,0

        loss = self.loss_func(pred, label)
        overall =  loss+0.1*loss_maps + 0.05*loss_type + 0.1*loss_mag
        if 'map' in pred_dict:
            synthesizer_loss,entropy_penalty=self.get_syn_loss(overall,pred_dict)
        else:
            synthesizer_loss, entropy_penalty = 0,0
        loss_dict = {
            'overall': overall,'synthesizer_loss':synthesizer_loss,'loss_type':loss_type,
            'loss_mag':loss_mag,'loss_maps':loss_maps,'entropy_penalty':entropy_penalty,
                     }
        return loss_dict

    def get_syn_loss(self, loss,pred_dict):
        entropy = pred_dict['entropy']
        log_prob = pred_dict['log_prob']
        normlized_lm=loss.detach()
        if log_prob is not None:
            self.optimizer_synthesizer.zero_grad()
            score_loss = torch.mean(-log_prob * normlized_lm)
            entropy_penalty = torch.mean(entropy)
            synthesizer_loss = score_loss - (1e-5) * entropy_penalty
            if synthesizer_loss.requires_grad:
                synthesizer_loss.backward()
                self.optimizer_synthesizer.step()
        else:
            synthesizer_loss=0
            entropy_penalty=0
        return synthesizer_loss,entropy_penalty
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']

        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def syn_preprocess(self,batch):
        imgs,lmks,msks,lbs=batch['image'].to(device),batch['landmark'].to(device),batch['mask'].to(device),batch['label'].to(device)
        half = len(imgs) // 2

        # imgs, lmks, msks, lbs = imgs[new_idx], lmks[new_idx], msks[new_idx], lbs[new_idx]

        img, fake_img, real_lmk, fake_lmk, real_mask, fake_mask,real_lb,fake_lb = \
           imgs[:half],imgs[half:],lmks[:half],lmks[half:],msks[:half],msks[half:],lbs[:half],lbs[half:]

        # conduct intentional real-fake switching to fit the stupid setting of original code.
        # TODO: too little number of 0. considering replacing it by taking real at simple aug. But many issues may raise
        switch_mask = torch.randint(0, 2, (img.shape[0],)).bool()
        img[switch_mask], fake_img[switch_mask], real_lmk[switch_mask], fake_lmk[switch_mask], real_mask[switch_mask], fake_mask[switch_mask], real_lb[switch_mask], fake_lb[switch_mask] = \
            fake_img[switch_mask],img[switch_mask],fake_lmk[switch_mask],real_lmk[switch_mask],fake_mask[switch_mask],real_mask[switch_mask],fake_lb[switch_mask],real_lb[switch_mask]

        log_prob, entropy, new_img, alt_mask, label, type_label, mag_label, mag_mask = \
        self.synthesizer(img, fake_img, real_lmk, fake_lmk, real_mask, fake_mask,label=lbs[:half])

        new_img = new_img.to(device)
        label = label.to(device)
        type_label = type_label.to(device)
        mag_label = mag_label.to(device)
        mag_mask = mag_mask.to(device)
        alt_mask = alt_mask.to(device)

        ################ simple augmentation  seems to be useless
        img_flip = torch.flip(new_img, (3,)).detach().clone()
        mask_flip = torch.flip(alt_mask, (3,)).detach().clone()
        new_img = torch.cat((new_img, img_flip))
        alt_mask = torch.cat((alt_mask, mask_flip))
        label = torch.cat((label, label))
        type_label = torch.cat((type_label, type_label))
        mag_label = torch.cat((mag_label, mag_label))
        mag_mask = torch.cat((mag_mask, mag_mask))
        return new_img,alt_mask,label,type_label,mag_label,mag_mask,log_prob, entropy


    def forward(self, data_dict: dict, inference=False) -> dict:
        if inference:
            new_img=data_dict['image']
            label=data_dict['label']
            features,map_fea = self.features(new_img)
            # get the prediction by classifier
            out,x = self.classifier(features)
            pred = out
            # get the probability of the pred
            prob = torch.softmax(pred, dim=1)[:, 1]
            pred_dict = {
                'cls': pred, 'prob': prob, 'feat': features,'label':label,
            }
        else:
            #print(data_dict['image'].device)
            new_img,alt_mask,label,type_label,mag_label,mag_mask,log_prob, entropy=self.syn_preprocess(data_dict)
            # get the features by backbone
            features,map_fea = self.features(new_img)
            # get the prediction by classifier
            out,x = self.classifier(features)
            map = self.backbone.estimateMap(map_fea)
            type=self.backbone.type_fc(x)
            mag=self.backbone.mag_fc(x)
            pred = out
            # get the probability of the pred
            prob = torch.softmax(pred, dim=1)[:, 1]

            # build the prediction dict for each output
            pred_dict = {
                'cls': pred, 'prob': prob, 'feat': features,'map':map,'type':type,'mag':mag, 'log_prob':log_prob,'label':label,
                'entropy':entropy,'alt_mask': alt_mask,'type_label':type_label,'mag_label':mag_label,'mag_mask':mag_mask
                         }
        return pred_dict

if __name__ == '__main__':
    with open(r'H:\code\DeepfakeBench\training\config\detector\sladd_xception.yaml', 'r') as f:
        config = yaml.safe_load(f)
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])
    detector=SLADDXceptionDetector(config=config).to(device)
    config['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config['sample_size']=256
    config['with_mask']=True
    config['with_landmark']=True
    config['use_data_augmentation']=True
    train_set = pairDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=32,
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
        continue
        imgs,lmks,msks=batch['image'].to(device),batch['landmark'].to(device),batch['mask'].to(device)
        batch['image'],batch['landmark'],batch['mask'], batch['label'] = \
            batch['image'].to(device), batch['landmark'].to(device), batch['mask'].to(device),batch['label'].to(device)
        half = len(imgs) // 2
        img, fake_img, real_lmk, fake_lmk, real_mask, fake_mask = imgs[:half],imgs[half:],lmks[:half],lmks[half:],msks[:half],msks[half:]

        predictions=detector(batch)
        losses = detector.get_losses(batch, predictions)
        optimizer.zero_grad()
        losses['overall'].backward()
        optimizer.step()

        if iteration > 10:
            break
