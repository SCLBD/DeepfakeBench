import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="jsloss")
class JS_Loss(AbstractLossClass):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        """
        Computes the Jensen-Shannon divergence loss.
        """
        # Compute the probability distributions
        inputs_prob = F.softmax(inputs, dim=1)
        targets_prob = F.softmax(targets, dim=1)

        # Compute the average probability distribution
        avg_prob = (inputs_prob + targets_prob) / 2

        # Compute the KL divergence component for each distribution
        kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        kl_inputs = kl_div_loss(inputs_prob.log(), avg_prob)
        kl_targets = kl_div_loss(targets_prob.log(), avg_prob)

        # Compute the Jensen-Shannon divergence
        loss = 0.5 * (kl_inputs + kl_targets)

        return loss