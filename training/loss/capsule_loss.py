import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="capsule_loss")
class CapsuleLoss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Computes the capsule loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the capsule loss.
        """
        # Compute the capsule loss
        loss_t = self.cross_entropy_loss(inputs[:,0,:], targets)

        for i in range(inputs.size(1) - 1):
            loss_t = loss_t + self.cross_entropy_loss(inputs[:,i+1,:], targets)
        return loss_t
