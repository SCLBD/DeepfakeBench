import torch
from torch import nn
from loss.abstract_loss_func import AbstractLossClass
from utils.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="classNseg_loss")
class ClassNsegLoss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.gamma = 1.0 # weight decay. default=5.0
        self.act_loss_fn = ActivationLoss()
        self.rect_loss_fn = ReconstructionLoss()
        self.seg_loss_fn = SegmentationLoss()

    def forward(self, inputs, targets):
        
        zero_abs = torch.abs(latent[:,0]).view(latent.shape[0], -1)
        zero = zero_abs.mean(dim=1)

        one_abs = torch.abs(latent[:,1]).view(latent.shape[0], -1)
        one = one_abs.mean(dim=1)
        
        loss_act = self.act_loss_fn(zero, one, labels_data)
        loss_act_data = loss_act.item()
    
        loss_seg = self.seg_loss_fn(seg, mask)
        loss_seg = loss_seg * self.gamma
        loss_seg_data = loss_seg.item()

        loss_rect = self.rect_loss_fn(rect, rgb)
        loss_rect = loss_rect * self.gamma
        loss_rect_data = loss_rect.item()
        loss_total = loss_act + loss_seg + loss_rect
        return loss_total, loss_act_data, loss_seg_data, loss_rect_data


class ActivationLoss(nn.Module):
    def __init__(self):
        super(ActivationLoss, self).__init__()

    def forward(self, zero, one, labels):

        loss_act = torch.abs(one - labels.data) + torch.abs(zero - (1.0 - labels.data))
        return 1 / labels.shape[0] * loss_act.sum()
        
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, reconstruction, groundtruth):

        return self.loss(reconstruction, groundtruth.data)

class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, segment, groundtruth):

        return self.loss(segment.view(segment.shape[0], segment.shape[1], segment.shape[2] * segment.shape[3]), 
            groundtruth.data.view(groundtruth.shape[0], groundtruth.shape[1] * groundtruth.shape[2]))
