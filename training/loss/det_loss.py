# import torch.nn as nn
# from loss.abstract_loss_func import AbstractLossClass
# from utils.registry import LOSSFUNC


# @LOSSFUNC.register_module(module_name="det_loss")
# class DETLoss(AbstractLossClass):
#     def __init__(self):
#         super().__init__()
#         self.loss_fn = nn.BCELoss()

#     def forward(self, inputs, targets):
#         """
#         Computes the bce loss.

#         Args:
#             inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
#             targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

#         Returns:
#             A scalar tensor representing the bce loss.
#         """
#         # Compute the bce loss
#         loss = self.loss_fn(inputs, targets.float())

#         return loss


# # loss init
#     det_criterion = MultiBoxLoss(
#         cfg['det_loss']['num_classes'],
#         cfg['det_loss']['overlap_thresh'],
#         cfg['det_loss']['prior_for_matching'],
#         cfg['det_loss']['bkg_label'],
#         cfg['det_loss']['neg_mining'],
#         cfg['det_loss']['neg_pos'],
#         cfg['det_loss']['neg_overlap'],
#         cfg['det_loss']['encode_target'],
#         cfg['det_loss']['use_gpu']
#     )
#     criterion = nn.CrossEntropyLoss()            

# loss_end_cls = criterion(outputs, labels)
# loss_l, loss_c = det_criterion(
#     (locations, confidence),
#     confidence_labels, location_labels
# )
# acc = sum(outputs.max(-1).indices == labels).item() / labels.shape[0]
# det_loss = 0.1 * (loss_l + loss_c)
# loss = det_loss + loss_end_cls
# loss.backward()


# class MultiBoxLoss(nn.Module):
#     """SSD Weighted Loss Function
#     Compute Targets:
#         1) Produce Confidence Target Indices by matching  ground truth boxes
#            with (default) 'priorboxes' that have jaccard index > threshold parameter
#            (default threshold: 0.5).
#         2) Produce localization target by 'encoding' variance into offsets of ground
#            truth boxes and their matched  'priorboxes'.
#         3) Hard negative mining to filter the excessive number of negative examples
#            that comes with using a large number of default bounding boxes.
#            (default negative:positive ratio 3:1)
#     Objective Loss:
#         L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#         Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
#         weighted by α which is set to 1 by cross val.
#         Args:
#             c: class confidences,
#             l: predicted boxes,
#             g: ground truth boxes
#             N: number of matched default boxes
#         See: https://arxiv.org/pdf/1512.02325.pdf for more details.
#     """

#     def __init__(self, num_classes, overlap_thresh, prior_for_matching,
#                  bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
#                  use_gpu=True):
#         super(MultiBoxLoss, self).__init__()
#         self.use_gpu = use_gpu
#         self.num_classes = num_classes
#         self.threshold = overlap_thresh
#         self.background_label = bkg_label
#         self.encode_target = encode_target
#         self.use_prior_for_matching = prior_for_matching
#         self.do_neg_mining = neg_mining
#         self.negpos_ratio = neg_pos
#         self.neg_overlap = neg_overlap
#         self.variance = [0.1, 0.2]  # cfg['variance']

#     # def forward(self, predictions, targets):
#     def forward(self, predictions, conf_t, loc_t):
#         """Multibox Loss
#         Args:
#             predictions (tuple): A tuple containing loc preds, conf preds,
#             and prior boxes from SSD net.
#                 conf shape: torch.size(batch_size,num_priors,num_classes)
#                 loc shape: torch.size(batch_size,num_priors,4)
#                 priors shape: torch.size(num_priors,4)

#             targets (tensor): Ground truth boxes and labels for a batch,
#                 shape: [batch_size,num_objs,5] (last idx is the label).
#         """
#         '''
#         priors = priors[:loc_data.size(1), :]
#         num_priors = (priors.size(0))
#         num_classes = self.num_classes

#         # match priors (default boxes) and ground truth boxes
#         loc_t = torch.Tensor(num, num_priors, 4)
#         conf_t = torch.LongTensor(num, num_priors)
#         for idx in range(num):
#             truths = targets[idx][:, :-1].data
#             labels = targets[idx][:, -1].data
#             defaults = priors.data
#             match(self.threshold, truths, defaults, self.variance, labels,
#                   loc_t, conf_t, idx)
#         '''
#         loc_data, conf_data = predictions
#         num = loc_data.size(0)
#         if self.use_gpu:
#             loc_t = loc_t.cuda()
#             conf_t = conf_t.cuda()
#         # wrap targets
#         loc_t = Variable(loc_t, requires_grad=False)
#         conf_t = Variable(conf_t, requires_grad=False)

#         pos = conf_t > 0
#         num_pos = pos.sum(dim=1, keepdim=True)

#         # Localization Loss (Smooth L1)
#         # Shape: [batch,num_priors,4]
#         pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
#         loc_p = loc_data[pos_idx].view(-1, 4)
#         loc_t = loc_t[pos_idx].view(-1, 4)
#         loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

#         # Compute max conf across batch for hard negative mining
#         batch_conf = conf_data.view(-1, self.num_classes)
#         loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

#         # Hard Negative Mining
#         # loss_c[pos] = 0  # filter out pos boxes for now
#         loss_c[pos.view(-1, 1)] = 0
#         loss_c = loss_c.view(num, -1)
#         _, loss_idx = loss_c.sort(1, descending=True)
#         _, idx_rank = loss_idx.sort(1)
#         num_pos = pos.long().sum(1, keepdim=True)
#         num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
#         neg = idx_rank < num_neg.expand_as(idx_rank)

#         # Confidence Loss Including Positive and Negative Examples
#         pos_idx = pos.unsqueeze(2).expand_as(conf_data)
#         neg_idx = neg.unsqueeze(2).expand_as(conf_data)
#         conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
#         targets_weighted = conf_t[(pos+neg).gt(0)]
#         loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

#         # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

#         N = num_pos.data.sum() if num_pos.data.sum() else 1
#         loss_l /= N
#         loss_c /= N
#         return loss_l, loss_c