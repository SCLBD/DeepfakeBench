from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.distributed as dist
import math


def l2_norm(input, axis=1):
    norm = torch.norm(input, p=2, dim=axis, keepdim=True)
    output = torch.div(input, norm)
    return output


def calc_logits(embeddings, kernel):
    """ calculate original logits
    """
    embeddings = l2_norm(embeddings, axis=1)
    kernel_norm = l2_norm(kernel, axis=0)
    cos_theta = torch.mm(embeddings, kernel_norm)
    cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
    with torch.no_grad():
        origin_cos = cos_theta.clone()
    return cos_theta, origin_cos


@torch.no_grad()
def all_gather_tensor(input_tensor):
    """ allgather tensor (difference size in 0-dim) from all workers
    """
    world_size = dist.get_world_size()

    tensor_size = torch.tensor([input_tensor.shape[0]], dtype=torch.int64).cuda()
    tensor_size_list = [torch.ones_like(tensor_size) for _ in range(world_size)]
    dist.all_gather(tensor_list=tensor_size_list, tensor=tensor_size, async_op=False)
    max_size = torch.cat(tensor_size_list, dim=0).max()

    padded = torch.empty(max_size.item(), *input_tensor.shape[1:], dtype=input_tensor.dtype).cuda()
    padded[:input_tensor.shape[0]] = input_tensor
    padded_list = [torch.ones_like(padded) for _ in range(world_size)]
    dist.all_gather(tensor_list=padded_list, tensor=padded, async_op=False)

    slices = []
    for ts, t in zip(tensor_size_list, padded_list):
        slices.append(t[:ts.item()])
    return torch.cat(slices, dim=0)


def calc_top1_acc(original_logits, label,ddp=False):
    """
    Compute the top1 accuracy during training
    :param original_logits: logits w/o margin, [bs, C]
    :param label: labels [bs]
    :return: acc in all gpus
    """
    assert (original_logits.size()[0] == label.size()[0])

    with torch.no_grad():
        _, max_index = torch.max(original_logits, dim=1, keepdim=False)  # local max logit
        count = (max_index == label).sum()
        if ddp:
            dist.all_reduce(count, dist.ReduceOp.SUM)

            return count.item() / (original_logits.size()[0] * dist.get_world_size())
        else:
            return count.item() / (original_logits.size()[0])

def l2_norm(input, axis=1):
    norm = torch.norm(input, p=2, dim=axis, keepdim=True)
    output = torch.div(input, norm)
    return output


class FC_ddp2(nn.Module):
    """
    Implement of  (CVPR2021 Consistent Instance False Positive Improves Fairness in Face Recognition)
    No model parallel is used
    """

    def __init__(self,
                 in_features,
                 out_features,
                 scale=64.0,
                 margin=0.4,
                 mode='cosface',
                 use_cifp=False,
                 reduction='mean',
                 ddp=False):
        """ Args:
            in_features: size of each input features
            out_features: size of each output features
            scale: norm of input feature
            margin: margin
        """
        super(FC_ddp2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features  # num of classes
        self.scale = scale
        self.margin = margin
        self.mode = mode
        self.use_cifp = use_cifp
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.ddp = ddp
        nn.init.normal_(self.kernel, std=0.01)

        self.criteria = torch.nn.CrossEntropyLoss(reduction=reduction)

    def apply_margin(self, target_cos_theta):
        assert self.mode in ['cosface', 'arcface'], 'Please check the mode'
        if self.mode == 'arcface':
            cos_m = math.cos(self.margin)
            sin_m = math.sin(self.margin)
            theta = math.cos(math.pi - self.margin)
            sinmm = math.sin(math.pi - self.margin) * self.margin
            sin_theta = torch.sqrt(1.0 - torch.pow(target_cos_theta, 2))
            cos_theta_m = target_cos_theta * cos_m - sin_theta * sin_m
            target_cos_theta_m = torch.where(
                target_cos_theta > theta, cos_theta_m, target_cos_theta - sinmm)
        elif self.mode == 'cosface':
            target_cos_theta_m = target_cos_theta - self.margin

        return target_cos_theta_m

    def forward(self, embeddings, label, return_logits=False):
        """

        :param embeddings: local gpu [bs, 512]
        :param label: local labels [bs]
        :param return_logits: bool
        :return:
        loss: computed local loss, w/wo CIFP
        acc: local accuracy in one gpu
        output: local logits with margins, with gradients, scaled, [bs, C].
        """
        sample_num = embeddings.size(0)

        if not self.use_cifp:
            cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
            target_cos_theta = cos_theta[torch.arange(0, sample_num), label].view(-1, 1)
            target_cos_theta_m = self.apply_margin(target_cos_theta)
            cos_theta.scatter_(1, label.view(-1, 1).long(), target_cos_theta_m)
        else:
            cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
            cos_theta_, _ = calc_logits(embeddings, self.kernel.detach())

            mask = torch.zeros_like(cos_theta)  # [bs，C]
            mask.scatter_(1, label.view(-1, 1).long(), 1.0)  # one-hot label / gt mask

            tmp_cos_theta = cos_theta - 2 * mask
            tmp_cos_theta_ = cos_theta_ - 2 * mask

            target_cos_theta = cos_theta[torch.arange(0, sample_num), label].view(-1, 1)
            target_cos_theta_ = cos_theta_[torch.arange(0, sample_num), label].view(-1, 1)

            target_cos_theta_m = self.apply_margin(target_cos_theta)

            far = 1 / (self.out_features - 1)  # ru+ value
            # far = 1e-5

            topk_mask = torch.greater(tmp_cos_theta, target_cos_theta)
            topk_sum = torch.sum(topk_mask.to(torch.int32))
            if self.ddp:
                dist.all_reduce(topk_sum)
            far_rank = math.ceil(far * (sample_num * (self.out_features - 1) * dist.get_world_size() - topk_sum))
            cos_theta_neg_topk = torch.topk((tmp_cos_theta - 2 * topk_mask.to(torch.float32)).flatten(),
                                            k=far_rank)[0]  # [far_rank]
            cos_theta_neg_topk = all_gather_tensor(cos_theta_neg_topk.contiguous())  # top k across all gpus
            cos_theta_neg_th = torch.topk(cos_theta_neg_topk, k=far_rank)[0][-1]

            cond = torch.mul(torch.bitwise_not(topk_mask), torch.greater(tmp_cos_theta, cos_theta_neg_th))
            cos_theta_neg_topk = torch.mul(cond.to(torch.float32), tmp_cos_theta)
            cos_theta_neg_topk_ = torch.mul(cond.to(torch.float32), tmp_cos_theta_)
            cond = torch.greater(target_cos_theta_m, cos_theta_neg_topk)

            cos_theta_neg_topk = torch.where(cond, cos_theta_neg_topk, cos_theta_neg_topk_)
            cos_theta_neg_topk = torch.pow(cos_theta_neg_topk, 2)  # F = z^p = cos^2
            times = torch.sum(torch.greater(cos_theta_neg_topk, 0).to(torch.float32), dim=1, keepdim=True)
            times = torch.where(torch.greater(times, 0), times, torch.ones_like(times))
            cos_theta_neg_topk = torch.sum(cos_theta_neg_topk, dim=1, keepdim=True) / times  # ri+/ru+

            target_cos_theta_m = target_cos_theta_m - (1 + target_cos_theta_) * cos_theta_neg_topk
            cos_theta.scatter_(1, label.view(-1, 1).long(), target_cos_theta_m)

        output = cos_theta * self.scale
        loss = self.criteria(output, label)
        acc = calc_top1_acc(origin_cos * self.scale, label,self.ddp)

        if return_logits:
            return loss, acc, output

        return loss, acc


class FC_ddp(nn.Module):
    """
    Implement of  (CVPR2021 Consistent Instance False Positive Improves Fairness in Face Recognition)
    No model parallel is used
    """

    def __init__(self,
                 in_features,
                 out_features,
                 scale=8.0,
                 margin=0.2,
                 mode='cosface',
                 use_cifp=False,
                 reduction='mean'):
        """ Args:
            in_features: size of each input features
            out_features: size of each output features
            scale: norm of input feature
            margin: margin
        """
        super(FC_ddp, self).__init__()
        self.in_features = in_features
        self.out_features = out_features  # num of classes
        self.scale = scale
        self.margin = margin
        self.mode = mode
        self.use_cifp = use_cifp
        # self.kernel = Parameter(torch.Tensor(in_features, out_features))
        # nn.init.normal_(self.kernel, std=0.01)

        self.criteria = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.sig = torch.nn.Sigmoid()

    def apply_margin(self, target_cos_theta):
        assert self.mode in ['cosface', 'arcface'], 'Please check the mode'
        if self.mode == 'arcface':
            cos_m = math.cos(self.margin)
            sin_m = math.sin(self.margin)
            theta = math.cos(math.pi - self.margin)
            sinmm = math.sin(math.pi - self.margin) * self.margin
            sin_theta = torch.sqrt(1.0 - torch.pow(target_cos_theta, 2))
            cos_theta_m = target_cos_theta * cos_m - sin_theta * sin_m
            target_cos_theta_m = torch.where(
                target_cos_theta > theta, cos_theta_m, target_cos_theta - sinmm)
        elif self.mode == 'cosface':
            target_cos_theta_m = target_cos_theta - self.margin

        return target_cos_theta_m

    def forward(self, embeddings, label, return_logits=False):
        """

        :param embeddings: local gpu [bs, 512]
        :param label: local labels [bs]
        :param return_logits: bool
        :return:
        loss: computed local loss, w/wo CIFP
        acc: local accuracy in one gpu
        output: local logits with margins, with gradients, scaled, [bs, C].
        """
        sample_num = embeddings.size(0)
        cos_theta = self.sig(embeddings)
        target_cos_theta = cos_theta[torch.arange(0, sample_num), label].view(-1, 1)
        # target_cos_theta_m = target_cos_theta - self.margin
        target_cos_theta = target_cos_theta - self.margin
        # cos_theta.scatter_(1, label.view(-1, 1).long(), target_cos_theta_m)
        out = cos_theta.clone()
        out.scatter_(1, label.view(-1, 1).long(), target_cos_theta)

        out = out * self.scale

        loss = self.criteria(out, label)

        return loss
