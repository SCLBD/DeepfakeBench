"""
# author: Kangran ZHAO
# email: kangranzhao@link.cuhk.edu.cn
# date: 2024-0410
# description: Class for the UIA-ViT Detector

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
@inproceedings{zhuang2020UIA,
  title={UIA-ViT: Unsupervised Inconsistency-Aware Method based on Vision Transformer for Face Forgery Detection},
  author={Zhuang, Wanyi and Chu, Qi and Tan, Zhentao and Liu, Qiankun and Yuan, Haojie and Miao, Changtao and Luo, Zixiang and Yu, Nenghai},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022},
}

Codes are modified based on GitHub repo https://github.com/wany0824/UIA-ViT
"""
from functools import partial

import torch
import torch.nn as nn
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
from sklearn.covariance import LedoitWolf
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .base_detector import AbstractDetector


@DETECTOR.register_module(module_name='uia_vit')
class UIAViTDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.batch_per_epoch = config["batch_per_epoch"]
        self.num_epoch = config["nEpochs"]

        self.batch_cnt = 0
        self.real_feature_list, self.fake_feature_list = [], []
        self.real_inv_covariance, self.fake_inv_covariance = None, None
        self.real_feature_mean, self.fake_feature_mean = None, None

        self.model = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.loss_weight = config["loss_func"]["weights"]

    def build_backbone(self, config):
        model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=2)
        state_dict = torch.hub.load_state_dict_from_url(config["pretrained"])
        del state_dict["head.bias"], state_dict["head.weight"]
        model.load_state_dict(state_dict, strict=False)

        return model

    def build_loss(self, config):
        cls_loss_class = LOSSFUNC[config["loss_func"]["cls_loss"]]
        pcl_loss_class = LOSSFUNC[config["loss_func"]["pcl_loss"]]
        cls_loss_func = cls_loss_class()
        pcl_loss_func = pcl_loss_class(c_real=self.model.c_real, c_fake=self.model.c_fake, c_cross=self.model.c_cross)

        return {"cls": cls_loss_func, "pcl": pcl_loss_func}

    def features(self, data_dict: dict) -> torch.tensor:
        pass

    def classifier(self, features: torch.tensor) -> torch.tensor:
        pass  # do not overwrite this, since classifier structure has been written in self.forward()

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        ce_loss = self.loss_func["cls"](pred, label)
        if self.batch_cnt > self.batch_per_epoch and self.model.training:
            pcl_loss = self.loss_func["pcl"](pred_dict["attention_map_real"],
                                             pred_dict["attention_map_fake"],
                                             pred_dict["feat"],
                                             self.real_feature_mean,
                                             self.real_inv_covariance,
                                             self.fake_feature_mean,
                                             self.fake_inv_covariance,
                                             data_dict["label"])
            overall_loss = ce_loss + \
                           self.loss_weight[0] * pcl_loss + \
                           self.loss_weight[1] * (1 / torch.abs(self.model.c_real) + 1 / torch.abs(self.model.c_fake)) + \
                           self.loss_weight[2] * torch.abs(self.model.c_cross)

            return {"overall": overall_loss, "ce_loss": ce_loss, "pcl_loss": pcl_loss,
                    "c1": self.model.c_real, "c2": self.model.c_fake, "c3": self.model.c_cross}
        else:
            return {"overall": ce_loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # compute MVG
        if self.model.training and self.batch_cnt != 0 and self.batch_cnt % (self.config["batch_per_epoch"] // 2) == 0:
            real_feature_tensor = torch.cat(self.real_feature_list, dim=0).cuda()
            self.real_inv_covariance = fit_inv_covariance(real_feature_tensor).cpu()
            self.real_feature_mean = real_feature_tensor.mean(dim=0).cpu()
            self.real_feature_list = []

            fake_feature_tensor = torch.cat(self.fake_feature_list, dim=0).cuda()
            self.fake_inv_covariance = fit_inv_covariance(fake_feature_tensor).cpu()
            self.fake_feature_mean = fake_feature_tensor.mean(dim=0).cpu()
            self.fake_feature_list = []

        step = self.batch_cnt / (self.batch_per_epoch * self.num_epoch) if self.model.training else 1
        pred, feature_patch, attention_map = self.model(data_dict["image"], step=step)

        # collect features of real patches and inner fake patches
        real_indices = torch.where(data_dict["label"] == 0.0)[0]
        feature_patch_real = feature_patch[real_indices[:4]]
        B, H, W, C = feature_patch_real.size()
        self.real_feature_list.append(feature_patch_real.reshape(B * H * W, C).cpu().detach())

        fake_indices = torch.where(data_dict["label"] == 1.0)[0]
        feature_patch_fake = feature_patch[fake_indices[:4], 3:11, 3:11, :]  # hard coding, extend config to modify if needed
        B, H, W, C = feature_patch_fake.size()
        self.fake_feature_list.append(feature_patch_fake.reshape(B * H * W, C).cpu().detach())

        attention_map_real = torch.sigmoid(torch.mean(attention_map[real_indices, :, 1:, 1:], dim=1))
        attention_map_fake = torch.sigmoid(torch.mean(attention_map[fake_indices, :, 1:, 1:], dim=1))

        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {"cls": pred,
                     "prob": prob,
                     "feat": feature_patch}

        del attention_map, feature_patch

        pred_dict["attention_map_real"] = attention_map_real
        pred_dict["attention_map_fake"] = attention_map_fake
        self.batch_cnt += 1

        return pred_dict


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # [B, 196, 768] -> [B, 196, 768*3] -> [B, 196, 3, 8, 96] -> [3, B, 8, 196, 96]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn_qk = (q @ k.transpose(-2, -1)) * self.scale
        attn_s = attn_qk.softmax(dim=-1)
        attn = self.attn_drop(attn_s)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_qk


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_attn, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        # f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, H*W, C]
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.c_real = nn.Parameter(torch.tensor(0.6))
        self.c_fake = nn.Parameter(torch.tensor(0.6))
        self.c_cross = nn.Parameter(torch.tensor(0.2))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.norm_middle = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, attn_blk, feat_blk=False):
        if feat_blk == False:
            feat_blk = attn_blk - 1
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if isinstance(attn_blk, int):
            for i, blk in enumerate(self.blocks):
                if i == feat_blk:
                    x_block = self.norm_middle(x)
                if i == attn_blk:
                    attn_block = attn
                x, attn = blk(x)
            x = self.norm(x)  # for vit_base_patch16_224: x.size() = [B, 14**2+1 (197) , 768]
            if i == attn_blk - 1:
                attn_block = attn
            if i == feat_blk - 1:
                x_block = x
        elif isinstance(attn_blk, list):
            attn_list = []
            for i, blk in enumerate(self.blocks):
                if i == feat_blk:
                    x_block = self.norm_middle(x)
                if i in attn_blk:
                    attn_list.append(attn)
                x, attn = blk(x)
            x = self.norm(x)  # for vit_base_patch16_224: x.size() = [B, 14**2+1 (197) , 768]
            if (i + 1) in attn_blk:
                attn_list.append(attn)
            if i == feat_blk - 1:
                x_block = x
            attn_block = torch.cat(attn_list, dim=1)

        x_block = x_block[:, 1:].reshape(
            (x_block.size(0), int(x_block.size(1) ** 0.5), int(x_block.size(1) ** 0.5), x_block.size(2)))
        return x, x_block, attn_block

    def forward(self, x, step=1, attn_blk=[8, 9, 10, 11, 12], feat_blk=6, k=12, thr=0.7, is_progressive=1):
        x, feat_block, attn_block = self.forward_features(x, attn_blk, feat_blk)

        x_cls, x_patch = x[:, 0], x[:, 1:]
        B, PP, C = x_patch.shape
        localization_map = torch.sigmoid(torch.mean(attn_block[:, :, 0, 1:], dim=1))

        if is_progressive:
            if step < 1 / 8.:
                localization_map = (torch.ones(B, 1, PP) / PP).to(x_patch.device)
            else:
                w = torch.sigmoid(torch.tensor(-k * (step - thr))).to(x_patch.device)
                localization_map = (w * torch.ones(B, 1, PP).to(x_patch.device) + (1 - w) * localization_map.reshape(B,
                                                                                                                     1,
                                                                                                                     PP).to(
                    x_patch.device)) / PP
        else:
            localization_map = localization_map.reshape(B, 1, PP).to(x_patch.device) / PP
        x = torch.cat([x_cls, torch.bmm(localization_map, x_patch).squeeze(1)], -1)
        x = self.head(x)
        return x, feat_block, attn_block


def fit_inv_covariance(samples):
    return torch.Tensor(LedoitWolf().fit(samples.cpu()).precision_).to(
        samples.device
    )
