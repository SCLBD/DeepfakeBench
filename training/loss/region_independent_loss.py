import torch
import torch.nn.functional as F
from detectors.multi_attention_detector import AttentionPooling
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="region_independent_loss")
class RegionIndependentLoss(AbstractLossClass):
    def __init__(self, M, N, alpha, alpha_decay, decay_batch, inter_margin, intra_margin):
        super().__init__()
        feature_centers = torch.zeros(M, N)
        self.register_buffer("feature_centers",
                             feature_centers.cuda() if torch.cuda.is_available() else feature_centers)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.decay_batch = decay_batch
        self.batch_cnt = 0
        self.inter_margin = inter_margin
        intra_margin = torch.Tensor(intra_margin)
        self.register_buffer("intra_margin", intra_margin.cuda() if torch.cuda.is_available() else intra_margin)
        self.atp = AttentionPooling()

    def forward(self, feature_maps_d, attention_maps, labels):
        B, N, H, W = feature_maps_d.size()
        B, M, AH, AW = attention_maps.size()
        if AH != H or AW != W:
            attention_maps = F.interpolate(attention_maps, (H, W), mode='bilinear', align_corners=True)
        feature_matrix = self.atp(feature_maps_d, attention_maps)

        # Calculate new feature centers. P.s., I don't know why to use no_grad() and detach() for so many times.
        feature_centers = self.feature_centers.detach()
        new_feature_centers = feature_centers + self.alpha * torch.mean(feature_matrix - feature_centers, dim=0)
        new_feature_centers = new_feature_centers.detach()
        with torch.no_grad():
            self.feature_centers = new_feature_centers

        # Calculate intra-class loss
        intra_margins = torch.gather(self.intra_margin.repeat(B, 1), dim=1, index=labels.unsqueeze(1))
        intra_class_loss = torch.mean(F.relu(torch.norm(feature_matrix - new_feature_centers, dim=-1) - intra_margins))

        # Calculate inter-class loss
        inter_class_loss = 0
        for i in range(M):
            for j in range(i + 1, M):
                inter_class_loss += F.relu(
                    self.inter_margin - torch.dist(new_feature_centers[i], new_feature_centers[j]), inplace=False)
        inter_class_loss = inter_class_loss / M / self.alpha

        # Count batch, this is used to simulate epoch, since alpha cannot be modified based on epoch due to code
        # structure. self.alpha should be modified every N batch.
        self.batch_cnt += 1
        if self.batch_cnt % self.decay_batch == 0:
            self.alpha *= self.alpha_decay

        return inter_class_loss + intra_class_loss
