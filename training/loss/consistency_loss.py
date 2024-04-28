import torch.nn as nn
import torch
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="consistency_loss")
class ConsistencyCos(nn.Module):
    def __init__(self):
        super(ConsistencyCos, self).__init__()
        # # CrossEntropy Loss
        # weight=torch.Tensor([4.0, 1.0])
        # if torch.cuda.is_available():
        #     weight = weight.cuda()
        # self.loss_fn = nn.CrossEntropyLoss(weight)
        self.loss_fn = nn.CrossEntropyLoss()
        self.mse_fn = nn.MSELoss()

    def forward(self, feat, inputs, targets):
        feat = nn.functional.normalize(feat, dim=1)
        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2): 2*int(feat.size(0)/2),:]

        cos = torch.einsum('nc,nc->n', [feat_0, feat_1]).unsqueeze(-1)
        labels = torch.ones((cos.shape[0],1), dtype=torch.float, requires_grad=False)
        if torch.cuda.is_available():
            labels = labels.cuda()
        self.consistency_rate = 1.0
        loss = self.consistency_rate * self.mse_fn(cos, labels) + self.loss_fn(inputs, targets)
        return loss

#
##FIXME to be implemented
class ConsistencyL2(nn.Module):
    def __init__(self):
        super(ConsistencyL2, self).__init__()
        self.mse_fn = nn.MSELoss()

    def forward(self, feat):
        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        loss = self.mse_fn(feat_0, feat_1)
        return loss

class ConsistencyL1(nn.Module):
    def __init__(self):
        super(ConsistencyL1, self).__init__()
        self.L1_fn = nn.L1Loss()

    def forward(self, feat):
        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        loss = self.L1_fn(feat_0, feat_1)
        return loss