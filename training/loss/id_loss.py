import torch
import torch.nn as nn
from loss.abstract_loss_func import AbstractLossClass
from utils.registry import LOSSFUNC

@LOSSFUNC.register_module(module_name="id_loss")
class IDLoss(AbstractLossClass):
    def __init__(self, margin=0.5):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.margin = margin

    def forward(self, x1, x2):
        cosine_similarity = self.cosine_similarity(x1, x2)
        theta = torch.acos(cosine_similarity)
        return 1 - torch.cos(theta + self.margin)