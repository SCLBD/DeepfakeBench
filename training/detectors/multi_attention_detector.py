"""
# author: Kangran ZHAO
# email: kangranzhao@link.cuhk.edu.cn
# date: 2024-0401
# description: Class for the Multi-attention Detector

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
@INPROCEEDINGS{9577592,
  author={Zhao, Hanqing and Wei, Tianyi and Zhou, Wenbo and Zhang, Weiming and Chen, Dongdong and Yu, Nenghai},
  booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title={Multi-attentional Deepfake Detection},
  year={2021},
  volume={},
  number={},
  pages={2185-2194},
  keywords={Measurement;Semantics;Feature extraction;Forgery;Pattern recognition;Feeds;Task analysis},
  doi={10.1109/CVPR46437.2021.00222}
  }

Codes are modified based on GitHub repo https://github.com/yoctta/multiple-attention
"""

import random

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
from networks import BACKBONE
from sklearn import metrics

from .base_detector import AbstractDetector


@DETECTOR.register_module(module_name='multi_attention')
class MultiAttentionDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_layer = {"b1": 1, "b2": 5, "b3": 9, "b4": 15, "b5": 21, "b6": 29, "b7": 31}
        self.mid_dim = config["mid_dim"]
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.batch_cnt = 0

        with torch.no_grad():
            layer_outputs = self.features({"image": torch.zeros(1, 3, config["resolution"], config["resolution"])})

        self.feature_layer = config["feature_layer"]
        self.attention_layer = config["attention_layer"]
        self.num_classes = config["backbone_config"]["num_classes"]
        self.num_shallow_features = layer_outputs[self.feature_layer].shape[1]
        self.num_attention_features = layer_outputs[self.attention_layer].shape[1]
        self.num_final_features = layer_outputs["final"].shape[1]
        self.num_attentions = config["num_attentions"]

        self.AGDA = AGDA(kernel_size=config["AGDA"]["kernel_size"],
                         dilation=config["AGDA"]["dilation"],
                         sigma=config["AGDA"]["sigma"],
                         threshold=config["AGDA"]["threshold"],
                         zoom=config["AGDA"]["zoom"],
                         scale_factor=config["AGDA"]["scale_factor"],
                         noise_rate=config["AGDA"]["noise_rate"])

        self.attention_generation = AttentionMap(self.num_attention_features, self.num_attentions)
        self.attention_pooling = AttentionPooling()
        self.texture_enhance = TextureEnhanceV1(self.num_shallow_features, self.num_attentions)  # Todo
        self.num_enhanced_features = self.texture_enhance.output_features
        self.num_features_d = self.texture_enhance.output_features_d
        self.projection_local = nn.Sequential(nn.Linear(self.num_attentions * self.num_enhanced_features, self.mid_dim),
                                              nn.Hardswish(),
                                              nn.Linear(self.mid_dim, self.mid_dim),
                                              nn.Hardswish())
        self.projection_final = nn.Sequential(nn.Linear(self.num_final_features, self.mid_dim),
                                              nn.Hardswish())
        self.ensemble_classifier_fc = nn.Sequential(nn.Linear(self.mid_dim * 2, self.mid_dim),
                                                    nn.Hardswish(),
                                                    nn.Linear(self.mid_dim, self.num_classes))
        self.dropout = nn.Dropout(config["dropout_rate"], inplace=True)
        self.dropout_final = nn.Dropout(config["dropout_rate_final"], inplace=True)

    def build_backbone(self, config):
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        model_config['pretrained'] = self.config.get('pretrained', None)
        backbone = backbone_class(model_config)

        return backbone

    def build_loss(self, config):
        cls_loss_class = LOSSFUNC[config["loss_func"]["cls_loss"]]
        ril_loss_class = LOSSFUNC[config["loss_func"]["ril_loss"]]
        cls_loss_func = cls_loss_class()
        ril_loss_func = ril_loss_class(M=config["num_attentions"],
                                       N=config["loss_func"]["ril_params"]["N"],
                                       alpha=config["loss_func"]["ril_params"]["alpha"],
                                       alpha_decay=config["loss_func"]["ril_params"]["alpha_decay"],
                                       decay_batch=config["batch_per_epoch"],
                                       inter_margin=config["loss_func"]["ril_params"]["inter_margin"],
                                       intra_margin=config["loss_func"]["ril_params"]["intra_margin"])

        return {"cls": cls_loss_func, "ril": ril_loss_func, "weights": config["loss_func"]["weights"]}

    def features(self, data_dict: dict) -> torch.tensor:
        x = data_dict["image"]
        layer_output = {}
        for name, module in self.backbone.efficientnet.named_children():
            if name == "_avg_pooling":
                layer_output["final"] = x
                break
            elif name != "_blocks":
                x = module(x)
            else:
                for i in range(len(module)):
                    x = module[i](x)
                    if i == self.block_layer["b1"]:
                        layer_output["b1"] = x
                    elif i == self.block_layer["b2"]:
                        layer_output["b2"] = x
                    elif i == self.block_layer["b3"]:
                        layer_output["b3"] = x
                    elif i == self.block_layer["b4"]:
                        layer_output["b4"] = x
                    elif i == self.block_layer["b5"]:
                        layer_output["b5"] = x
                    elif i == self.block_layer["b6"]:
                        layer_output["b6"] = x
                    elif i == self.block_layer["b7"]:
                        layer_output["b7"] = x

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        layer_output["logit"] = self.backbone.last_layer(x)

        return layer_output

    def classifier(self, features: torch.tensor) -> torch.tensor:
        pass  # do not overwrite this, since classifier structure has been written in self.forward()

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        if self.batch_cnt <= self.config["backbone_nEpochs"] * self.config["batch_per_epoch"]:
            label = data_dict["label"]
            pred = pred_dict["cls"]
            ce_loss = self.loss_func["cls"](pred, label)

            return {"overall": ce_loss, "ce_loss": ce_loss}
        else:
            label = data_dict["label"]
            pred = pred_dict["cls"]
            feature_maps_d = pred_dict["feature_maps_d"]
            attention_maps = pred_dict["attentions"]

            ce_loss = self.loss_func["cls"](pred, label)
            ril_loss = self.loss_func["ril"](feature_maps_d, attention_maps, label)
            weights = self.loss_func["weights"]
            over_all_loss = weights[0] * ce_loss + weights[1] * ril_loss

            return {"overall": over_all_loss, "ce_loss": ce_loss, "ril_loss": ril_loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        return metric_batch_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        self.batch_cnt += 1
        if self.batch_cnt <= self.config["backbone_nEpochs"] * self.config["batch_per_epoch"]:
            layer_output = self.features(data_dict)
            pred = layer_output["logit"]
            prob = torch.softmax(pred, dim=1)[:, 1]
            pred_dict = {"cls": pred,
                         "prob": prob,
                         "feat": layer_output["final"]}

        else:
            if not inference:  # use AGDA when training
                with torch.no_grad():
                    layer_output = self.features(data_dict)
                    raw_attentions = layer_output[self.attention_layer]
                    attention_maps = self.attention_generation(raw_attentions)
                    data_dict["image"], _ = self.AGDA.agda(data_dict["image"], attention_maps)

            # Get Attention Maps
            layer_output = self.features(data_dict)
            raw_attentions = layer_output[self.attention_layer]
            attention_maps = self.attention_generation(raw_attentions)

            # Get Textural Feature Matrix P
            shallow_features = layer_output[self.feature_layer]
            enhanced_features, feature_maps_d = self.texture_enhance(shallow_features, attention_maps)
            textural_feature_matrix_p = self.attention_pooling(enhanced_features, attention_maps)
            B, M, N = textural_feature_matrix_p.size()
            feature_matrix = self.dropout(textural_feature_matrix_p).view(B, -1)
            feature_matrix = self.projection_local(feature_matrix)

            # Get Global Feature G
            final = layer_output["final"]
            attention_maps2 = attention_maps.sum(dim=1, keepdim=True)  # [B, 1, H_A, W_A]
            final = self.attention_pooling(final, attention_maps2, norm=1).squeeze(1)  # [B, C_F]
            final = self.projection_final(final)
            final = F.hardswish(final)

            # Get the Prediction by Ensemble Classifier
            feature_matrix = torch.cat((feature_matrix, final), dim=1)  # [B, 2 * mid_dim]
            pred = self.ensemble_classifier_fc(feature_matrix)  # [B, 2]

            # Get probability
            prob = torch.softmax(pred, dim=1)[:, 1]

            pred_dict = {"cls": pred,
                         "prob": prob,
                         "feat": layer_output["final"],
                         "attentions": attention_maps,
                         "feature_maps_d": feature_maps_d}

        return pred_dict


class AttentionMap(nn.Module):
    def __init__(self, in_channels, num_attention):
        super(AttentionMap, self).__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        self.mask[0, 0, 2:-2, 2:-2] = 1
        self.num_attentions = num_attention
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, num_attention, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_attention)

    def forward(self, x):
        """
        Convert deep feature to attention map
        Args:
            x: extracted features
        Returns:
            attention_maps: conventionally 4 attention maps
        """
        if self.num_attentions == 0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)

        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x) + 1
        mask = F.interpolate(self.mask, (x.shape[2], x.shape[3]), mode='nearest')

        return x * mask


class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, attentions, norm=2):
        """
        Bilinear Attention Pooing, when used for
        Args:
            features: [Tensor in [B, C_F, H_F, W_F]] extracted feature maps, either shallow ones or deep ones ???
            attentions: [Tensor in [B, M, H, W]] attention maps, conventionally 4 attention maps (M = 4)
            norm: [int, default=2] 1 for deep features, 2 for shallow features
        Returns:
            feature_matrix: [Tensor in [B, M, C_F] or [B, M, 1]] P (shallow feature) or G (deep feature) ???
        """
        feature_size = features.size()[-2:]
        attention_size = attentions.size()[-2:]
        if feature_size != attention_size:
            attentions = F.interpolate(attentions, size=feature_size, mode='bilinear', align_corners=True)

        if len(features.shape) == 4:
            # In TextureEnhanceV1, in accordance with paper
            feature_matrix = torch.einsum('imjk,injk->imn', attentions, features)  # [B, M, C_F]
        else:
            # In TextureEnhanceV2
            feature_matrix = torch.einsum('imjk,imnjk->imn', attentions, features)

        if norm == 1:  # Used for deep feature BAP
            w = torch.sum(attentions + 1e-8, dim=(2, 3)).unsqueeze(-1)
            feature_matrix /= w
        elif norm == 2:  # Used for shallow feature BAP
            feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)

        return feature_matrix


class TextureEnhanceV1(nn.Module):
    def __init__(self, num_features, num_attentions):
        super().__init__()
        # self.output_features=num_features
        self.output_features = num_features * 4
        self.output_features_d = num_features
        self.conv0 = nn.Conv2d(num_features, num_features, 1)
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features * 2, num_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * num_features)
        self.conv3 = nn.Conv2d(num_features * 3, num_features, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(3 * num_features)
        self.conv_last = nn.Conv2d(num_features * 4, num_features * 4, 1)
        self.bn4 = nn.BatchNorm2d(4 * num_features)
        self.bn_last = nn.BatchNorm2d(num_features * 4)

    def forward(self, feature_maps, attention_maps=(1, 1)):
        """
        Texture Enhancement Block V1, in accordance with description in paper
        1. Local average pooling.
        2. Residual local features.
        3. Dense Net
        Args:
            feature_maps: [Tensor in [B, C', H', W']] extracted shallow features
            attention_maps: [Tensor in [B, M, H_A, W_A]] calculated attention maps, or
                            [Tuple with two float elements] local average grid scale,
                            used for conduct local average pooling, local patch size is decided by attention map size.
        Returns:
            feature_maps: [Tensor in [B, C_1, H_1, W_1]] enhanced feature maps
            feature_maps_d: [Tensor in [B, C', H_A, W_A]] textural information

        """
        B, N, H, W = feature_maps.shape
        if type(attention_maps) == tuple:
            attention_size = (int(H * attention_maps[0]), int(W * attention_maps[1]))
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)
        feature_maps = feature_maps - F.interpolate(feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]),
                                                    mode='nearest')
        feature_maps0 = self.conv0(feature_maps)
        feature_maps1 = self.conv1(F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = torch.cat([feature_maps0, feature_maps1], dim=1)
        feature_maps2 = self.conv2(F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = torch.cat([feature_maps1_, feature_maps2], dim=1)
        feature_maps3 = self.conv3(F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = torch.cat([feature_maps2_, feature_maps3], dim=1)
        feature_maps = self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True)))
        return feature_maps, feature_maps_d


class TextureEnhanceV2(nn.Module):
    def __init__(self, num_features, num_attentions):
        super().__init__()
        self.output_features = num_features
        self.output_features_d = num_features
        self.conv_extract = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv0 = nn.Conv2d(num_features * num_attentions, num_features * num_attentions, 5, padding=2,
                               groups=num_attentions)
        self.conv1 = nn.Conv2d(num_features * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn1 = nn.BatchNorm2d(num_features * num_attentions)
        self.conv2 = nn.Conv2d(num_features * 2 * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn2 = nn.BatchNorm2d(2 * num_features * num_attentions)
        self.conv3 = nn.Conv2d(num_features * 3 * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn3 = nn.BatchNorm2d(3 * num_features * num_attentions)
        self.conv_last = nn.Conv2d(num_features * 4 * num_attentions, num_features * num_attentions, 1,
                                   groups=num_attentions)
        self.bn4 = nn.BatchNorm2d(4 * num_features * num_attentions)
        self.bn_last = nn.BatchNorm2d(num_features * num_attentions)

        self.M = num_attentions

    def cat(self, a, b):
        B, C, H, W = a.shape
        c = torch.cat([a.reshape(B, self.M, -1, H, W), b.reshape(B, self.M, -1, H, W)], dim=2).reshape(B, -1, H, W)
        return c

    def forward(self, feature_maps, attention_maps=(1, 1)):
        """
        Args:
            feature_maps: [Tensor in [B, N, H, W]] extracted feature maps from shallow layer
            attention_maps: [Tensor in [B, M, H_A, W_A] or float of (H_ratio, W_ratio)] either extracted attention maps
                or average pooling down-sampling ratio
        Returns:
            feature_maps, feature_maps_d: [Tensor in [B, M, N, H, W], Tensor in [B, N, H, W]] feature maps after dense
                network and non-textural feature map D
        """
        B, N, H, W = feature_maps.shape
        if type(attention_maps) == tuple:
            attention_size = (int(H * attention_maps[0]), int(W * attention_maps[1]))
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
        feature_maps = self.conv_extract(feature_maps)
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)
        if feature_maps.size(2) > feature_maps_d.size(2):
            feature_maps = feature_maps - F.interpolate(feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]),
                                                        mode='nearest')
        attention_maps = (
            torch.tanh(F.interpolate(attention_maps.detach(), (H, W), mode='bilinear', align_corners=True))).unsqueeze(
            2) if type(attention_maps) != tuple else 1
        feature_maps = feature_maps.unsqueeze(1)
        feature_maps = (feature_maps * attention_maps).reshape(B, -1, H, W)
        feature_maps0 = self.conv0(feature_maps)
        feature_maps1 = self.conv1(F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = self.cat(feature_maps0, feature_maps1)
        feature_maps2 = self.conv2(F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = self.cat(feature_maps1_, feature_maps2)
        feature_maps3 = self.conv3(F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = self.cat(feature_maps2_, feature_maps3)
        feature_maps = F.relu(self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True))),
                              inplace=True)
        feature_maps = feature_maps.reshape(B, -1, N, H, W)
        return feature_maps, feature_maps_d


class AGDA(nn.Module):
    def __init__(self, kernel_size, dilation, sigma, threshold, zoom, scale_factor, noise_rate):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sigma = sigma
        self.noise_rate = noise_rate
        self.scale_factor = scale_factor
        self.threshold = threshold
        self.zoom = zoom
        self.filter = kornia.filters.GaussianBlur2d((self.kernel_size, self.kernel_size), (self.sigma, self.sigma))

    def mod_func(self, x):
        threshold = random.uniform(*self.threshold) if type(self.threshold) == list else self.threshold
        zoom = random.uniform(*self.zoom) if type(self.zoom) == list else self.zoom
        bottom = torch.sigmoid((torch.tensor(0.) - threshold) * zoom)

        return (torch.sigmoid((x - threshold) * zoom) - bottom) / (1 - bottom)

    def soft_drop2(self, x, attention_map):
        with torch.no_grad():
            attention_map = self.mod_func(attention_map)
            B, C, H, W = x.size()
            xs = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
            xs = self.filter(xs)
            xs += torch.randn_like(xs) * self.noise_rate
            xs = F.interpolate(xs, (H, W), mode='bilinear', align_corners=True)
            x = x * (1 - attention_map) + xs * attention_map
        return x

    def agda(self, X, attention_map):
        with torch.no_grad():
            attention_weight = torch.sum(attention_map, dim=(2, 3))
            attention_map = F.interpolate(attention_map, (X.size(2), X.size(3)), mode="bilinear", align_corners=True)
            attention_weight = torch.sqrt(attention_weight + 1)
            index = torch.distributions.categorical.Categorical(attention_weight).sample()
            index1 = index.view(-1, 1, 1, 1).repeat(1, 1, X.size(2), X.size(3))
            attention_map = torch.gather(attention_map, 1, index1)
            atten_max = torch.max(attention_map.view(attention_map.shape[0], 1, -1), 2)[0] + 1e-8
            attention_map = attention_map / atten_max.view(attention_map.shape[0], 1, 1, 1)

            return self.soft_drop2(X, attention_map), index
