import torch
from metrics.registry import LOSSFUNC
from .abstract_loss_func import AbstractLossClass


def mahalanobis_distance(values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor) -> torch.Tensor:
    """Compute the batched mahalanobis distance.

    values is a batch of feature vectors.
    mean is either the mean of the distribution to compare, or a second
    batch of feature vectors.
    inv_covariance is the inverse covariance of the target distribution.
    """
    assert values.dim() == 2
    assert 1 <= mean.dim() <= 2
    assert inv_covariance.dim() == 2
    assert values.shape[1] == mean.shape[-1]
    assert mean.shape[-1] == inv_covariance.shape[0]
    assert inv_covariance.shape[0] == inv_covariance.shape[1]

    if mean.dim() == 1:  # Distribution mean.
        mean = mean.unsqueeze(0)
    x_mu = values - mean  # batch x features
    # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
    dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)

    return dist.sqrt()


@LOSSFUNC.register_module(module_name="patch_consistency_loss")
class PatchConsistencyLoss(AbstractLossClass):
    def __init__(self, c_real, c_fake, c_cross):
        super().__init__()
        self.c_real = c_real
        self.c_fake = c_fake
        self.c_cross = c_cross

    def forward(self, attention_map_real, attention_map_fake, feature_patch, real_feature_mean, real_inv_covariance,
                fake_feature_mean, fake_inv_covariance, labels):
        # calculate mahalanobis distance
        B, H, W, C = feature_patch.size()
        dist_real = mahalanobis_distance(feature_patch.reshape(B * H * W, C), real_feature_mean.cuda(),
                                         real_inv_covariance.cuda())
        dist_fake = mahalanobis_distance(feature_patch.reshape(B * H * W, C), fake_feature_mean.cuda(),
                                         fake_inv_covariance.cuda())
        fake_indices = torch.where(labels == 1.0)[0]
        index_map = torch.relu(dist_real - dist_fake).reshape((B, H, W))[fake_indices, :]

        # loss for real samples
        if attention_map_real.shape[0] == 0:
            loss_real = 0
        else:
            B, PP, PP = attention_map_real.shape
            c_matrix = (1 - self.c_real) * torch.eye(PP).cuda() + self.c_real * torch.ones(PP).cuda()
            c_matrix = c_matrix.expand(B, -1, -1)
            loss_real = torch.sum(torch.abs(attention_map_real - c_matrix)) / (B * (PP * PP - PP))

        if attention_map_fake.shape[0] == 0:
            loss_fake = 0
        else:
            B, PP, PP = attention_map_fake.shape
            c_matrix = []
            for b in range(B):
                fake_indices = torch.where(index_map[b].reshape(-1) > 0)[0]
                real_indices = torch.where(index_map[b].reshape(-1) <= 0)[0]
                tmp = torch.zeros((PP, PP)).cuda() + self.c_cross
                for i in fake_indices:
                    tmp[i, fake_indices] = self.c_fake
                for i in real_indices:
                    tmp[i, real_indices] = self.c_real
                c_matrix.append(tmp)

            c_matrix = torch.stack(c_matrix).cuda()
            loss_fake = torch.sum(torch.abs(attention_map_fake - c_matrix)) / (B * (PP * PP - PP))

        return loss_real + loss_fake
