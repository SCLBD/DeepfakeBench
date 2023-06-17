import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

import argparse

from config import config, update_config
from networks.cls_hrnet import get_cls_net


from utils.plot_utils import plot_FaceMask

net = 'cls_HR'


class FaceXRay(nn.Module):
    def __init__(self, cfg, choice=net):
        super(FaceXRay, self).__init__()
        if choice == 'cls_HR':
            self.convnet = get_cls_net(cfg)
            self.post_process = nn.Sequential(
                nn.Conv2d(
                    in_channels=270,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.Upsample(size=(256, 256), mode='bilinear',
                            align_corners=True),
                nn.Sigmoid()
            )

            saved = torch.load('./pretrained/hrnet_w18.pth', map_location='cpu')
            self.convnet.load_state_dict(saved)
            print('Load HRnet')

        else:
            raise NotImplementedError(choice)

        self.classifier = nn.Linear(128*128, 2)

        self.init_weight()

    def forward(self, x):
        fea = self.convnet.features(x)
        # print(fea.size())
        mask = self.post_process(fea)

        score = F.adaptive_avg_pool2d(mask, 128)
        feat = score.view(score.size(0), -1)
        score = self.classifier(feat)

        return mask, score, feat  # w feat

    def init_weight(self):
        for ly in self.post_process.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.xavier_normal_(ly.weight.data, gain=0.02)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

        nn.init.xavier_normal_(self.classifier.weight.data, gain=0.02)


model_name = 'face-xray_{}_metrics'.format(net)


class Model():
    """
    wrapper for different detection method.
    """

    def __init__(self, opt, logdir=None, train=True):
        if opt is not None:
            self.meta = opt.meta

            update_config(config, opt)

        self.model = FaceXRay(config)

        self.cls_criterion = nn.CrossEntropyLoss()
        self.mask_criterion = nn.BCELoss()

        self.train = train
        if train:
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr,
                                        betas=(opt.beta1, 0.999), weight_decay=0.0005)

        if logdir is not None:
            # tensor board writer
            timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            log = '{}/{}/{}'.format(logdir, model_name, self.meta)
            log = log + '_{}'.format(timenow)
            print('TensorBoard log dir: {}'.format(log))

            self.writer = SummaryWriter(log_dir=log)
#        print(self.model)

    def speed_up(self, cuda, ngpu):
        if cuda:
            self.model = self.model.cuda()
            self.mask_criterion = self.mask_criterion.cuda()
            self.cls_criterion = self.cls_criterion.cuda()

        if(ngpu > 1):
            self.model = DataParallel(self.model, range(ngpu))
            self.mask_criterion = DataParallel(
                self.mask_criterion, range(ngpu))
            self.cls_criterion = DataParallel(self.cls_criterion, range(ngpu))

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            print('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def load_ckpt_cuda(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.module.load_state_dict(saved.state_dict())
            else:
                self.model.module.load_state_dict(saved)
            print('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, dataset, epoch, iters, best=False):
        save_dir = "./ckpts"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        mid_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)

        sub_dir = os.path.join(mid_dir, self.meta)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        subsub_dir = os.path.join(sub_dir, dataset)
        if not os.path.exists(subsub_dir):
            os.mkdir(subsub_dir)

        if best:
            ckpt_name = "epoch_{}_iter_{}_best.pth".format(epoch, iters)
        else:
            ckpt_name = "epoch_{}_iter_{}.pth".format(epoch, iters)

        save_path = os.path.join(subsub_dir, ckpt_name)

        torch.save(self.model.module.state_dict(), save_path)

        print("Checkpoint saved to {}".format(save_path))

    def optimize(self, inp, label):
        img, gt_mask = inp
        mask, score, feat = self.model(img)
        # label = label.unsqueeze(1)

        # 1. classification loss
        # loss_cls = self.cls_criterion(cls1, lab1)
        # loss_cls += self.cls_criterion(cls2, lab2)
        loss_cls = self.cls_criterion(score, label).mean()

        loss_mask = self.mask_criterion(mask, gt_mask.float()).mean()

        w_mask = 100.0
        loss = w_mask*loss_mask + loss_cls

        if self.train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # embeddings = face_f1, bg_f1, face_f2, bg_f2
        return score, (loss_cls, loss_mask), mask, feat

    def inference(self, img, label):
        mask, score, _ = self.model(img)
        # label = label.unsqueeze(1)

        # 1. classification loss
        # loss_cls = self.cls_criterion(cls1, lab1)
        # loss_cls += self.cls_criterion(cls2, lab2)
        loss_cls = self.cls_criterion(score, label).mean()

        return score, loss_cls, mask

    def update_tensorboard(self, inputs,  loss, step, label, acc=None, score=None, datas=None, name='train'):
        assert self.writer
        img, gt_mask, mask = inputs

        # 1. triplet loss
        if loss is not None:
            loss_cls, loss_mask = loss
            loss_dic = {
                'Cls': loss_cls,
                'Mask': loss_mask,
            }
            self.writer.add_scalars('{}/Loss'.format(name),
                                    tag_scalar_dict=loss_dic, global_step=step)

        # 2. acc
        if acc is not None:
            self.writer.add_scalar('{}/Acc'.format(name),
                                   acc, global_step=step)

        # 3. mask
        if gt_mask is None:
            gt_mask = torch.zeros_like(mask)
        self.writer.add_figure('Mask/{}'.format(name),
                               plot_FaceMask(
                                   img.detach(), gt_mask.detach(), mask.detach(), score.detach(), label.detach()),
                               global_step=step)
        # 4. PR-curve
        if datas is not None:
            self.writer.add_pr_curve(
                name,
                labels=datas[:,1].long(), predictions=datas[:, 0], global_step=step)

    def update_tensorboard_test_accs(self, accs, step, feas=None, label=None, name='test'):
        assert self.writer
        if isinstance(accs, list):
            self.writer.add_scalars('{}/ACC'.format(name),
                                    tag_scalar_dict=accs[0], global_step=step)
            self.writer.add_scalars('{}/AUC'.format(name),
                                    tag_scalar_dict=accs[1], global_step=step)
            self.writer.add_scalars('{}/EER'.format(name),
                                    tag_scalar_dict=accs[2], global_step=step)
            self.writer.add_scalars('{}/AP'.format(name),
                                    tag_scalar_dict=accs[3], global_step=step)
        else:
            self.writer.add_scalars('{}/AUC'.format(name),
                                    tag_scalar_dict=accs, global_step=step)



if __name__ == "__main__":
    model = Model(opt=None, train=False)
    B, C, H, W = 10, 3, 256, 256
    dummy = torch.rand((B, C, H, W))
    mask = torch.rand((B, 1, H, W))
    label = torch.ones(B)

    score, loss, mm = model.optimize((dummy, mask), label)

    print('mask size:', mm.size())
    # loss = torch.sum(score)
    # loss.backward()
