"""
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the XceptionDetector

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
@inproceedings{rossler2019faceforensics++,
  title={Faceforensics++: Learning to detect manipulated facial images},
  author={Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1--11},
  year={2019}
}
"""

import logging
from collections import OrderedDict

import clip
import math
import numpy as np
import torch
import torch.nn as nn
from detectors import DETECTOR
from einops import rearrange
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
from sklearn import metrics

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='sta_clip')
class StACLIPDetector(AbstractDetector):
    def __init__(self, config, demo=None):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = I3DHead(
            num_classes=2,
            in_channels=1024,
            spatial_type='avg',
            dropout_ratio=0.5
        )
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        assert self.config['resolution'] == 224, 'The resolution of the input image should be 224x224'
        # assert self.config['clip_size'] == 8, 'The number of frames should be 8'
        # prepare the backbone
        backbone = ViT_CLIP(
            input_resolution=224,
            num_frames=self.config['clip_size'],
            patch_size=14,
            width=1024,
            layers=14,
            heads=16,
            drop_path_rate=0.1,
            num_tadapter=1,
            adapter_scale=0.5,
            pretrained=True
        )

        # ## freeze some parameters
        # for name, param in backbone.named_parameters():
        #     if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name:
        #         param.requires_grad = False

        # for name, param in backbone.named_parameters():
        #     print('{}: {}'.format(name, param.requires_grad))
        # num_param = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        # num_total_param = sum(p.numel() for p in backbone.parameters())
        # print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label.long())
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self):
        y_pred, y_true = self.video_calculation(self.video_names, self.prob, self.label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label

        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}

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


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t h w c')
        x = self.norm(x)
        x = rearrange(x, 'b t h w c -> b c t h w')
        return x


class DepthwiseConv3D(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               groups=in_channels,
                               padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2))
        self.bn1 = nn.BatchNorm3d(num_features=in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               groups=in_channels,
                               padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2))
        self.bn2 = nn.BatchNorm3d(num_features=in_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               groups=in_channels,
                               padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2))
        self.bn3 = nn.BatchNorm3d(num_features=in_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        return x


# class ViT_Adapter(nn.Module):
#     def __init__(self, num_frames=8, in_channels=1024, out_channels=1024):
#         super().__init__()
#         self.num_frames=num_frames
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.adapter_channels = int(1024 * 0.5)

#         self.down = nn.Linear(in_features=self.in_channels, out_features=self.adapter_channels)
#         self.gelu1 = nn.GELU()

#         self.s_conv = DepthwiseConv3D(in_channels=self.adapter_channels, kernel_size=(1, 3, 3))
#         self.t_conv = DepthwiseConv3D(in_channels=self.adapter_channels, kernel_size=(3, 1, 1))

#         self.gelu = nn.GELU()

#         self.up = nn.Linear(in_features=self.adapter_channels, out_features=self.out_channels)
#         self.gelu2 = nn.GELU()

#     def forward(self, x):
#         # hw+1 bt c
#         n, bt, c = x.shape
#         H = round(math.sqrt(n - 1))
#         x_in = x

#         x = self.down(x)
#         x = self.gelu1(x)

#         cls = x[0, :, :].unsqueeze(0)
#         x = x[1:, :, :]

#         x = rearrange(x, '(h w) (b t) c -> b c t h w', t=self.num_frames, h=H)

#         # Apply depthwise 3D convolutions
#         xs = self.s_conv(x)
#         xt = self.t_conv(x)

#         # Fusion of xs and xt
#         x = (xs + xt) / 2
#         x = self.gelu(x)
#         x = rearrange(x, 'b c t h w -> (h w) (b t) c')

#         x = torch.cat([cls, x], dim=0)

#         x = self.up(x)
#         x = self.gelu2(x)

#         # residual
#         x += x_in
#         return x


class ViT_Adapter(nn.Module):
    def __init__(self, num_frames=8, in_channels=1024, out_channels=1024):
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adapter_channels = int(1024 * 0.5)

        self.down = nn.Linear(in_features=self.in_channels, out_features=self.adapter_channels)
        self.gelu1 = nn.GELU()

        self.s_conv1 = DepthwiseConv3D(in_channels=self.adapter_channels, kernel_size=(1, 3, 3))
        self.s_conv2 = DepthwiseConv3D(in_channels=self.adapter_channels, kernel_size=(1, 5, 5))
        self.s_conv3 = DepthwiseConv3D(in_channels=self.adapter_channels, kernel_size=(1, 7, 7))

        self.t_conv1 = DepthwiseConv3D(in_channels=self.adapter_channels, kernel_size=(3, 1, 1))
        self.t_conv2 = DepthwiseConv3D(in_channels=self.adapter_channels, kernel_size=(5, 1, 1))
        self.t_conv3 = DepthwiseConv3D(in_channels=self.adapter_channels, kernel_size=(7, 1, 1))

        self.gelu = nn.GELU()

        self.up = nn.Linear(in_features=self.adapter_channels, out_features=self.out_channels)
        self.gelu2 = nn.GELU()

        self.cross_attention = CrossAttention(embed_dim=self.adapter_channels, num_heads=2)

    def forward(self, x):
        # hw+1 bt c
        n, bt, c = x.shape
        H = round(math.sqrt(n - 1))
        x_in = x

        x = self.down(x)
        x = self.gelu1(x)

        cls = x[0, :, :].unsqueeze(0)
        x = x[1:, :, :]

        x = rearrange(x, '(h w) (b t) c -> b c t h w', t=self.num_frames, h=H)

        # Apply depthwise 3D convolutions
        xs1 = self.s_conv1(x)
        xs2 = self.s_conv2(x)
        xs3 = self.s_conv3(x)

        xt1 = self.t_conv1(x)
        xt2 = self.t_conv2(x)
        xt3 = self.t_conv3(x)

        # Fusion of xs and xt with residual connections
        xs = (xs1 + xs2 + xs3) / 3 + x
        xt = (xt1 + xt2 + xt3) / 3 + x

        # cross attention
        xs, xt = self.cross_attention(xs, xt)

        x = (xs + xt) / 2
        x = self.gelu(x)
        x = rearrange(x, 'b c t h w -> (h w) (b t) c')

        x = torch.cat([cls, x], dim=0)

        x = self.up(x)
        x = self.gelu2(x)

        # residual
        x += x_in
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.spatial_to_temporal = nn.MultiheadAttention(embed_dim, num_heads)
        self.temporal_to_spatial = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, spatial, temporal):
        # B, C, T, H, W
        B, C, T, H, W = spatial.shape

        # Flatten the spatial and temporal dimensions
        spatial = spatial.view(B, C, T, H * W)  # [B, C, T, H*W]
        temporal = temporal.view(B, C, T, H * W)  # [B, C, T, H*W]

        # Permute to [T*H*W, B, C] for MultiheadAttention
        spatial = spatial.permute(2, 0, 3, 1).reshape(T * H * W, B, C)  # [T*H*W, B, C]
        temporal = temporal.permute(2, 0, 3, 1).reshape(T * H * W, B, C)  # [T*H*W, B, C]

        # Apply cross attention
        s2t, _ = self.spatial_to_temporal(temporal, spatial, spatial)
        t2s, _ = self.temporal_to_spatial(spatial, temporal, temporal)

        # Reshape back to original dimensions
        s2t = s2t.view(T, H * W, B, C).permute(2, 3, 0, 1).reshape(B, C, T, H, W)
        t2s = t2s.view(T, H * W, B, C).permute(2, 3, 0, 1).reshape(B, C, T, H, W)

        return s2t, t2s


class ResidualAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 1024
        n_head = 16
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.Adapter = ViT_Adapter()

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x):
        # x shape [HW+1, BT, C]
        x = x + self.attention(self.ln_1(x))
        # x = self.Adapter(x)
        x = x + self.mlp(self.ln_2(x))
        x = self.Adapter(x)
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1.,
                 drop_path=0.1):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock() for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ViT_CLIP(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int,
                 drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale,
                                       drop_path=drop_path_rate)

        self.ln_post = LayerNorm(width)

        # self.init_weights()

    def init_weights(self):
        logger.info(f'load model from: {self.pretrained}')
        # Load OpenAI CLIP pretrained weights
        clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
        pretrain_dict = clip_model.visual.state_dict()
        del clip_model
        del pretrain_dict['proj']
        msg = self.load_state_dict(pretrain_dict, strict=False)
        logger.info('Missing keys: {}'.format(msg.missing_keys))
        logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        torch.cuda.empty_cache()
        # zero-initialize Adapters
        for n1, m1 in self.named_modules():
            if 'Adapter' in n1:
                for n2, m2 in m1.named_modules():
                    if 'up' in n2:
                        logger.info('init:  {}.{}'.format(n1, n2))
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def extract_class_indices(self, labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    def get_feat(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        # n = h*w+1
        n = x.shape[1]

        x = rearrange(x, '(b t) n c -> (b n) t c', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t c -> (b t) n c', n=n)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        return x

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x = self.get_feat(x)

        x = x[:, 0]
        x = rearrange(x, '(b t) d -> b d t', b=B, t=T)

        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x


class I3DHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score


if __name__ == '__main__':
    vit_model = ViT_CLIP(
        input_resolution=224,
        num_frames=8,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        drop_path_rate=0.1,
        num_tadapter=1,
        adapter_scale=0.5,
        pretrained=True
    )

    i3d_head = I3DHead(
        num_classes=2,
        in_channels=768,
        spatial_type='avg',
        dropout_ratio=0.5
    )

    rand_input = torch.rand(2, 8, 3, 224, 224)
    feat = vit_model(rand_input)
    print(feat.shape)
    output = i3d_head(feat)
    print(output.shape)
