'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the SRMDetector

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
@inproceedings{luo2021generalizing,
  title={Generalizing face forgery detection with high-frequency features},
  author={Luo, Yuchen and Zhang, Yong and Yan, Junchi and Liu, Wei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={16317--16326},
  year={2021}
}

Notes:
Other implementation modules are provided by the authors.
'''

import os
import datetime
import numbers
import math
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
import random

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='srm')
class SRMDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # prepare the backbone for rgb and srm branch
        self.backbone_rgb = self.build_backbone(config)
        self.backbone_srm = self.build_backbone(config)

        # srm specific layers and modules
        self.noise = GaussianNoise(clip=1)
        self.blur = GaussianSmoothing(channels=3, kernel_size=7, sigma=0.8)
        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)
        self.relu = nn.ReLU(inplace=True)
        self.att_map = None
        self.srm_sa = SRMPixelAttention(3)
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dual_cma0 = DualCrossModalAttention(in_dim=728)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728)
        self.fusion = FeatureFusionModule()

        # prepare the loss function
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        assert config['backbone_name'] == 'xception', "SRM only supports the xception backbone"
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        # To get a good performance, use the ImageNet-pretrained Xception model
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model from {}'.format(config['pretrained']))
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class(gamma=0., m=0.45, s=30, t=1.)  # use am-softmax for srm, params are specified by the author
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        x = data_dict['image']  # get the image as input for srm
        srm = self.srm_conv0(x)

        x = self.backbone_rgb.fea_part1_0(x)
        y = self.backbone_srm.fea_part1_0(srm) \
            + self.srm_conv1(x)
        y = self.relu(y)

        x = self.backbone_rgb.fea_part1_1(x)
        y = self.backbone_srm.fea_part1_1(y) \
            + self.srm_conv2(x)
        y = self.relu(y)

        # srm guided spatial attention
        self.att_map = self.srm_sa(srm)
        x = x * self.att_map + x   # use the residual
        x = self.srm_sa_post(x)

        x = self.backbone_rgb.fea_part2(x)
        y = self.backbone_srm.fea_part2(y)

        x, y = self.dual_cma0(x, y)

        x = self.backbone_rgb.fea_part3(x)
        y = self.backbone_srm.fea_part3(y)

        x, y = self.dual_cma1(x, y)

        x = self.backbone_rgb.fea_part4(x)
        y = self.backbone_srm.fea_part4(y)

        x = self.backbone_rgb.fea_part5(x)
        y = self.backbone_srm.fea_part5(y)

        fea = self.fusion(x, y)
        return fea
    
    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone_rgb.classifier(features)
    
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
        # we dont compute the video-level metrics for training
        self.video_names = []
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


# ===================================== other modules for SRM # =====================================


class SRMConv2d(nn.Module):

    def __init__(self, learnable=False):
        super(SRMConv2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(30, 3, 5, 5), 
                                requires_grad=learnable)
        self.bias = nn.Parameter(torch.Tensor(30), \
                              requires_grad=learnable)
        self.reset_parameters()

    def reset_parameters(self):
        SRM_npy = np.load('lib/component/SRM_Kernels.npy')
        # print(SRM_npy.shape)
        SRM_npy = np.repeat(SRM_npy, 3, axis=1)
        # print(SRM_npy.shape)
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, stride=1, padding=2)    


class SRMConv2d_simple(nn.Module):
    
    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
        # filter3：hor 2rd
        # filter3 = [[0, 0, 0, 0, 0],
        #            [0, 0, 1, 0, 0],
        #            [0, 1, -4, 1, 0],
        #            [0, 0, 1, 0, 0],
        #            [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],#, filter1, filter1],
                   [filter2],#, filter2, filter2],
                   [filter3]]#, filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)    # (3,3,5,5)
        return filters


class SRMConv2d_Separate(nn.Module):
    
    def __init__(self, inc, outc, learnable=False):
        super(SRMConv2d_Separate, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(3*inc, outc, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        for ly in self.out_conv.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

    def forward(self, x):
        '''
        x: imgs (Batch,inc, H, W)
        kernel: (outc,inc,kH,kW)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=self.inc)
        out = self.truc(out)
        out = self.out_conv(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
        # filter3：hor 2rd
        # filter3 = [[0, 0, 0, 0, 0],
        #            [0, 0, 1, 0, 0],
        #            [0, 1, -4, 1, 0],
        #            [0, 0, 1, 0, 0],
        #            [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],#, filter1, filter1],
                   [filter2],#, filter2, filter2],
                   [filter3]]#, filter3, filter3]]  # (3,3,5,5)  =>  (3,1,5,5)
        filters = np.array(filters)
        # filters = np.repeat(filters, inc, axis=1)
        filters = np.repeat(filters, inc, axis=0)
        filters = torch.FloatTensor(filters)    # (3*inc,1,5,5)
        # print(filters.size())
        return filters


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma=0.1, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.kernel_size = kernel_size
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
                    dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        if self.training:
            return self.conv(input, weight=self.weight, groups=self.groups, padding=self.kernel_size//2)
        else:
            return input


class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.1, clip=1):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std
        self.clip = clip

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(self.mean, self.std)
            return torch.clamp(x + noise, -self.clip, self.clip)
        else:
            return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CrossModalAttention(nn.Module):
    """ CMA attention Layer"""

    def __init__(self, in_dim, activation=None, ratio=8, cross_value=True):
        super(CrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.cross_value = cross_value

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        proj_query = self.query_conv(x).view(
            B, -1, H*W).permute(0, 2, 1)  # B , HW, C
        proj_key = self.key_conv(y).view(
            B, -1, H*W)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW
        attention = self.softmax(energy)  # BX (N) X (N)
        if self.cross_value:
            proj_value = self.value_conv(y).view(
                B, -1, H*W)  # B , C , HW
        else:
            proj_value = self.value_conv(x).view(
                B, -1, H*W)  # B , C , HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma*out + x

        if self.activation is not None:
            out = self.activation(out)

        return out  # , attention


class DualCrossModalAttention(nn.Module):
    """ Dual CMA attention Layer"""

    def __init__(self, in_dim, activation=None, size=16, ratio=8, ret_att=False):
        super(DualCrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.ret_att = ret_att

        # query conv
        self.key_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv_share = nn.Conv2d(
            in_channels=in_dim//ratio, out_channels=in_dim//ratio, kernel_size=1)
        
        self.linear1 = nn.Linear(size*size, size*size)
        self.linear2 = nn.Linear(size*size, size*size)

        # separated value conv
        self.value_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.value_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        def _get_att(a, b):
            proj_key1 = self.key_conv_share(self.key_conv1(a)).view(
                B, -1, H*W).permute(0, 2, 1)  # B , HW, C
            proj_key2 = self.key_conv_share(self.key_conv2(b)).view(
                B, -1, H*W)  # B X C x (*W*H)
            #print('proj_key1:', proj_key1[0][0][:5].cpu().detach().numpy())
            #print('proj_key2:', proj_key2[0][:5][0:5].cpu().detach().numpy())
            energy = torch.bmm(proj_key1, proj_key2)  # B, HW, HW
            #print('energy:', energy[0][0][:5].cpu().detach().numpy())
            attention1 = self.softmax(self.linear1(energy))
            attention2 = self.softmax(self.linear2(energy.permute(0,2,1)))  # BX (N) X (N)
            #print('1:', attention1[0]==attention1[1])
            #print('2:', attention2[0]==attention2[1])

            return attention1, attention2
        
        att_y_on_x, att_x_on_y = _get_att(x, y)       
        #print('att_y_on_x:', att_y_on_x[0][0][:5].cpu().detach().numpy()) 
        proj_value_y_on_x = self.value_conv2(y).view(
            B, -1, H*W)  # B , C , HW       
        out_y_on_x = torch.bmm(proj_value_y_on_x, att_y_on_x.permute(0, 2, 1))
        out_y_on_x = out_y_on_x.view(B, C, H, W)
        out_x = self.gamma1*out_y_on_x + x
        
        proj_value_x_on_y = self.value_conv1(x).view(
            B, -1, H*W)  # B , C , HW       
        out_x_on_y = torch.bmm(proj_value_x_on_y, att_x_on_y.permute(0, 2, 1))
        out_x_on_y = out_x_on_y.view(B, C, H, W)
        out_y = self.gamma2*out_x_on_y + y

        if self.ret_att:
            return out_x, out_y, att_y_on_x, att_x_on_y
        
        return out_x, out_y  # , attention


class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SRMPixelAttention, self).__init__()
        self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pa = SpatialAttention()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_srm = self.srm(x)
        fea = self.conv(x_srm)
        # fea += fea * self.ca(fea)
        att_map = self.pa(fea)
        # return x * y
        return att_map


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048*2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        #fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)      # Is it correct? F *(1+a) or  F * a?
        fuse_fea = fuse_fea * self.ca(fuse_fea)    # changed by yong 
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)
