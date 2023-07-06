import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from networks.xception_sladd import TransferModel
from networks.syn import synthesizer
from loss.am_softmax import AMSoftmaxLoss

model_name = 'sladd_40rames'

class Model():
    """
    wrapper for different detection method.
    """

    def __init__(self, opt, logdir=None, train=True):
        if opt is not None:
            self.meta = opt.meta
            self.opt = opt
            self.ngpu = opt.ngpu

        self.writer = None
        self.logdir = logdir
        dropout = 0.5
        self.model = TransferModel('xception', dropout=dropout, return_fea=True)
        self.synthesizer = synthesizer()
        self.cls_criterion = AMSoftmaxLoss(gamma=0., m=0.45, s=30, t=1.)
        self.train = train
        self.l1loss = nn.MSELoss()
        params = ([p for p in self.model.parameters()])
        params_synthesizer = ([p for p in self.synthesizer.parameters()])
        if train:
            self.optimizer = optim.Adam(params,lr=opt.lr,betas=(opt.beta1, 0.999),
                                        weight_decay=opt.weight_decay)
            self.optimizer_synthesizer = optim.Adam(params_synthesizer, lr=opt.lr/4, betas=(opt.beta1, 0.999),
                                        weight_decay=opt.weight_decay)


    def define_summary_writer(self):
        if self.logdir is not None:
            # tensor board writer
            timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            log = '{}/{}/{}'.format(self.logdir, model_name, self.meta)
            log = log + '_{}'.format(timenow)
            print('TensorBoard log dir: {}'.format(log))

            self.writer = SummaryWriter(log_dir=log)
            
    def speed_up(self, cuda, ngpu):
        if cuda:
            self.model.cuda()
            self.synthesizer.cuda()
            self.cls_criterion = self.cls_criterion.cuda() # criterion to gpu
            self.l1loss = self.l1loss.cuda()

        if(ngpu > 1):
            self.model = nn.DataParallel(self.model, range(ngpu))
            # self.mask_criterion = DataParallel(
            # self.mask_criterion, range(ngpu))
            self.cls_criterion = nn.DataParallel(self.cls_criterion, range(ngpu))

    def setTrain(self):
        self.model.train()
        self.synthesizer.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path=None, synthesizer_path=0):
        if model_path !=0 and os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            print('Model found in {}'.format(model_path))

        if synthesizer_path != 0 and os.path.isfile(synthesizer_path):
            saved = torch.load(synthesizer_path, map_location='cpu')
            suffix = synthesizer_path.split('.')[-1]
            if suffix == 'p':
                self.synthesizer.load_state_dict(saved.state_dict())
            else:
                self.synthesizer.load_state_dict(saved)
            print('synthesizer found in {}'.format(model_path))

    def save_ckpt(self, dataset, epoch, iters, save_dir, best=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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
            ckpt_name2 = "epoch_{}_iter_{}_best_syn.pth".format(epoch, iters)
        else:
            ckpt_name = "epoch_{}_iter_{}.pth".format(epoch, iters)
            ckpt_name2 = "epoch_{}_iter_{}_syn.pth".format(epoch, iters)

        save_path = os.path.join(subsub_dir, ckpt_name)
        save_path_ctrl = os.path.join(subsub_dir, ckpt_name2)

        if self.ngpu > 1:
            torch.save(self.model.module.state_dict(), save_path)
            torch.save(self.synthesizer.module.state_dict(), save_path_ctrl)
        else:
            torch.save(self.model.state_dict(), save_path)
            torch.save(self.synthesizer.state_dict(), save_path_ctrl)

        print("Checkpoint saved to {}".format(save_path))

    def optimize(self, img, label, fake_img, real_lmk, fake_lmk, gt_mask, fake_mask, epoch):
        log_prob = None
        device = torch.device("cuda")
        gt_mask = gt_mask.to(device)
        img = img.to(device)
        fake_img = fake_img.to(device)
        log_prob, entropy, new_img, alt_mask, label, type_label, mag_label, mag_mask = \
            self.synthesizer(img, fake_img, real_lmk, fake_lmk, gt_mask, fake_mask, label)
        new_img = new_img.to(device)
        label = label.to(device)
        type_label = type_label.to(device)
        mag_label = mag_label.to(device)
        mag_mask = mag_mask.to(device)
        alt_mask = alt_mask.to(device)
        ################ simple augmentation
        img_flip = torch.flip(new_img, (3,)).detach().clone()
        mask_flip = torch.flip(alt_mask, (3,)).detach().clone()
        new_img = torch.cat((new_img, img_flip))
        alt_mask = torch.cat((alt_mask, mask_flip))
        label = torch.cat((label, label))
        type_label = torch.cat((type_label, type_label))
        mag_label = torch.cat((mag_label, mag_label))
        mag_mask = torch.cat((mag_mask, mag_mask))
        ################
        ret = self.model(new_img)

        score, fea, map, type, mag = ret
        mag = mag.view(-1)

        if fea is not None:
            del(fea)
        loss_cls = self.cls_criterion(score, label).mean()
        loss_type = self.cls_criterion(type, type_label).mean()
        loss_mag = self.l1loss(mag*mag_mask, mag_label*mag_mask).mean()
        loss_maps = self.l1loss(map, alt_mask)
        loss = loss_cls + 0.1*loss_maps + 0.05*loss_type + 0.1*loss_mag

        if self.train:
            self.optimizer_synthesizer.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if log_prob is not None:
                lm = loss.detach()
                normlized_lm = lm
                score_loss = torch.mean(-log_prob * normlized_lm)
                entropy_penalty = torch.mean(entropy)
                synthesizer_loss = score_loss - (1e-5) * entropy_penalty
                synthesizer_loss.backward()
                self.optimizer_synthesizer.step()

        return label, (score, loss)

    def inference(self, img, label):
        with torch.no_grad():
            ret = self.model(img)
            score, fea, map, type, mag = ret
            loss_cls = self.cls_criterion(score, label).mean()
            return score, loss_cls

    def update_tensorboard(self, loss, step, acc=None, datas=None, name='train'):
        assert self.writer
        if loss is not None:
            loss_dic = {'Cls': loss}
            self.writer.add_scalars('{}/Loss'.format(name), tag_scalar_dict=loss_dic,
                                    global_step=step)

        if acc is not None:
            self.writer.add_scalar('{}/Acc'.format(name), acc, global_step=step)

        if datas is not None:
            self.writer.add_pr_curve(name, labels=datas[:, 1].long(),
                                     predictions=datas[:, 0], global_step=step)

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

        if feas is not None:
            metadata = []
            mat = None
            for key in feas:
                for i in range(feas[key].size(0)):
                    lab = 'fake' if label[key][i] == 1 else 'real'
                    metadata.append('{}_{:02d}_{}'.format(key, int(i), lab))
                if mat is None:
                    mat = feas[key]
                else:
                    mat = torch.cat((mat, feas[key]), dim=0)

            self.writer.add_embedding(mat, metadata=metadata, label_img=None,
                                      global_step=step, tag='default')