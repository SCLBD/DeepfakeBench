import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="bce")
class BCELoss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, inputs, targets):
        """
        Computes the bce loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the bce loss.
        """
        # Compute the bce loss
        loss = self.loss_fn(inputs, targets.float())

        return loss