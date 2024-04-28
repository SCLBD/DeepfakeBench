# --------------------------------------------------------
# Two Stream Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Hangyan Jiang
# --------------------------------------------------------

# Testing part
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse


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


if __name__ == "__main__":
    im = cv2.imread('E:\SRM\component\FF-F2F_0.png')
    im_ten = im/255*2-1
    im_ten = torch.from_numpy(im_ten).unsqueeze(0).permute(0, 3, 1, 2).float()
    # im_ten = torch.cat((im_ten, im_ten), dim=1)
    srm_conv = SRMConv2d_simple(inc=3)
    srm_conv1 = SRMConv2d_Separate(inc=3, outc=3)

    srm = srm_conv(im_ten)
    print(srm.size())

    def t2im(t):

        # t = (t+1)/2*255
        t = t*255
        im = t.squeeze().detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        return im

    cv2.imshow('ori', im)
    cv2.imshow('srm', t2im(srm))
    cv2.imshow('srm1', t2im(srm_conv1(im_ten)))
    # cv2.imshow('srm2', t2im(srm_conv(srm)))

    cv2.waitKey()
    

