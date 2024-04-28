InPlaceABN = None
from torch import nn
import torch.nn.functional as F


class Conv3dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm3d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, skip_channels, out_channels, use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LightDecoderBlock(nn.Module):
    def __init__(
        self, in_channels, skip_channels, out_channels, use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


def freeze_net(model: nn.Module, freeze_prefixs):
    flag = False
    for name, param in model.named_parameters():
        items = name.split(".")
        if items[0] == "module":
            prefix = items[1]
        else:
            prefix = items[0]
        if prefix in freeze_prefixs:
            if param.requires_grad is True:
                param.requires_grad = False
            flag = True
            # print("freeze",name)

    assert flag


def unfreeze_net(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = True


from .resnet_helper import ResBlock, get_trans_func


class ResDecoderBlock(nn.Module):
    def __init__(
        self, in_channels, skip_channels, out_channels, use_batchnorm=True,
    ):
        super().__init__()
        trans_func = get_trans_func("bottleneck_transform")
        self.conv1 = ResBlock(
            in_channels + skip_channels,
            out_channels,
            3,
            1,
            trans_func,
            out_channels//2,
            num_groups=1,
            stride_1x1=False,
            inplace_relu=True,
            eps=1e-5,
            bn_mmt=0.1,
            dilation=1,
            norm_module=nn.BatchNorm3d,
        )

        self.conv2 = ResBlock(
            out_channels,
            out_channels,
            3,
            1,
            trans_func,
            out_channels//2,
            num_groups=1,
            stride_1x1=False,
            inplace_relu=True,
            eps=1e-5,
            bn_mmt=0.1,
            dilation=1,
            norm_module=nn.BatchNorm3d,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
