import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from efficientnet_pytorch import EfficientNet
from utils.registry import BACKBONE


@BACKBONE.register_module(module_name="efficientnetb4")
class EfficientNetB4(nn.Module):
    def __init__(self, efficientnetb4_config):
        super(EfficientNetB4, self).__init__()
        """ Constructor
        Args:
            efficientnetb4_config: configuration file with the dict format
        """
        self.num_classes = efficientnetb4_config["num_classes"]
        inc = efficientnetb4_config["inc"]
        self.dropout = efficientnetb4_config["dropout"]
        self.mode = efficientnetb4_config["mode"]

        # Load the EfficientNet-B4 model without pre-trained weights
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4')
        # self.efficientnet = EfficientNet.from_name('efficientnet-b4')

        # Modify the first convolutional layer to accept input tensors with 'inc' channels
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)

        # Remove the last layer (the classifier) from the EfficientNet-B4 model
        self.efficientnet._fc = nn.Identity()

        if self.dropout:
            # Add dropout layer if specified
            self.dropout_layer = nn.Dropout(p=self.dropout)

        # Initialize the last_layer layer
        self.last_layer = nn.Linear(1792, self.num_classes)

        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(1792, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def features(self, x):
        # Extract features from the EfficientNet-B4 model
        x = self.efficientnet.extract_features(x)
        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        return x

    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Apply dropout if specified
        if self.dropout:
            x = self.dropout_layer(x)

        # Apply last_layer layer
        x = self.last_layer(x)
        return x

    def forward(self, x):
        # Extract features and apply classifier layer
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def init_weights(self, pretrained: Union[bool, str] = False):
        if isinstance(pretrained, str):
            self.efficientnet.load_state_dict(torch.load(pretrained), strict=False)
        elif pretrained:
            self.efficientnet.load_state_dict(torch.hub.load_state_dict_from_url(
                f'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b4-6ed6700e.pth'))
            # If inc is not 3, initialize the first convolutional layer with normal distribution
            if self.efficientnet._conv_stem.in_channels != 3:
                nn.init.kaiming_normal_(self.efficientnet._conv_stem.weight, mode='fan_out', nonlinearity='relu')
                if self.efficientnet._conv_stem.bias is not None:
                    nn.init.zeros_(self.efficientnet._conv_stem.bias)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

