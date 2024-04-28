import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from PIL import Image
import cv2
from torchvision import transforms as T
from skimage import measure
from skimage.transform import PiecewiseAffineTransform, warp
from torch.autograd import Variable
from scipy.ndimage import binary_erosion, binary_dilation

from dataset.pair_dataset import pairDataset
from dataset.utils.color_transfer import color_transfer
from dataset.utils.faceswap_utils_sladd import blendImages as alpha_blend_fea
from dataset.utils import faceswap



class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:  # whether the number of filters grows first
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class SeparableConv2d(nn.Module):
  def __init__(self, c_in, c_out, ks, stride=1, padding=0, dilation=1, bias=False):
    super(SeparableConv2d, self).__init__()
    self.c = nn.Conv2d(c_in, c_in, ks, stride, padding, dilation, groups=c_in, bias=bias)
    self.pointwise = nn.Conv2d(c_in, c_out, 1, 1, 0, 1, 1, bias=bias)

  def forward(self, x):
    x = self.c(x)
    x = self.pointwise(x)
    return x

class Xception_SLADDSyn(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=2, num_region=7, num_type=2, num_mag=1, inc=6):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_SLADDSyn, self).__init__()
        self.num_region = num_region
        self.num_type = num_type
        self.num_mag = num_mag
        dropout = 0.5

        # Entry flow
        self.iniconv = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        # self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc_region = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(2048, num_region))
        self.fc_type = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(2048, num_type))
        self.fc_mag = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(2048, num_mag))

    def fea_part1_0(self, x):
        x = self.iniconv(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def fea_part1_1(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part1(self, x):
        x = self.iniconv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        return x

    def fea_part4(self, x):
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        return x

    def fea_part5(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x

    def features(self, input):
        x = self.fea_part1(input)

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)

        x = self.fea_part5(x)
        return x

    def classifier(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out, x

    def forward(self, input):
        x = self.features(input)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        region_num = self.fc_region(x)
        type_num = self.fc_type(x)
        mag = self.fc_mag(x)

        return region_num, type_num, mag


def mask_postprocess(mask):
    def blur_mask(mask):
        blur_k = 2 * np.random.randint(1, 10) - 1

        # kernel = np.ones((blur_k+1, blur_k+1), np.uint8)
        # mask = cv2.erode(mask, kernel)

        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

        return mask

    # random erode/dilate
    prob = np.random.rand()
    if prob < 0.3:
        erode_k = 2 * np.random.randint(1, 10) + 1
        kernel = np.ones((erode_k, erode_k), np.uint8)
        mask = cv2.erode(mask, kernel)
    elif prob < 0.6:
        erode_k = 2 * np.random.randint(1, 10) + 1
        kernel = np.ones((erode_k, erode_k), np.uint8)
        mask = cv2.dilate(mask, kernel)

    # random blur
    if np.random.rand() < 0.9:
        mask = blur_mask(mask)

    return mask

def xception(num_region=7, num_type=2, num_mag=1, pretrained='imagenet', inc=6):
    model = Xception_SLADDSyn(num_region=num_region, num_type=num_type, num_mag=num_mag, inc=inc)
    return model



class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """

    def __init__(self, config, num_region=7, num_type=2, num_mag=1, return_fea=False, inc=6):
        super(TransferModel, self).__init__()
        self.return_fea = return_fea
        def return_pytorch04_xception(pretrained=True):
            # Raises warning "src not broadcastable to dst" but thats fine
            model = xception(num_region=num_region, num_type=num_type, num_mag=num_mag, inc=inc, pretrained=False)
            if pretrained:
                # Load model in torch 0.4+
                # model.fc = model.last_linear
                # del model.last_linear
                state_dict = torch.load(config['pretrained'])
                print('Loaded pretrained model (ImageNet)....')
                for name, weights in state_dict.items():
                    if 'pointwise' in name:
                        state_dict[name] = weights.unsqueeze(
                            -1).unsqueeze(-1)
                model.load_state_dict(state_dict, strict=False)
                # model.last_linear = model.fc
                # del model.fc
            return model

        self.model = return_pytorch04_xception()
        # Replace fc

        if inc != 3:
            self.model.iniconv = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
            nn.init.xavier_normal(self.model.iniconv.weight.data, gain=0.02)

    def set_trainable_up_to(self, boolean=False, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on lib, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise NotImplementedError('Layer not found, cant finetune!'.format(
                    layername))
        else:
            # Make fc trainable
            for param in self.model.last_linear.parameters():
                param.requires_grad = True

    def forward(self, x):
        region_num, type_num, mag = self.model(x)
        return region_num, type_num, mag

    def features(self, x):
        x = self.model.features(x)
        return x

    def classifier(self, x):
        out, x = self.model.classifier(x)
        return out, x



def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_random_mask(mask, res=256):
    randwl = np.random.randint(10, 60)
    randwr = np.random.randint(10, 60)
    randhu = np.random.randint(10, 60)
    randhd = np.random.randint(10, 60)
    newmask = np.zeros(mask.shape)
    mask = np.where(mask > 0.1, 1, 0)
    props = measure.regionprops(mask)
    if len(props) == 0:
        return newmask
    center_x, center_y = props[0].centroid
    center_x = int(round(center_x))
    center_y = int(round(center_y))
    newmask[max(center_x - randwl, 0):min(center_x + randwr, res - 1),
    max(center_y - randhu, 0):min(center_x + randhd, res - 1)] = 1
    newmask *= mask
    return newmask


def random_deform(mask, nrows, ncols, mean=0, std=10):
    h, w = mask.shape[:2]
    rows = np.linspace(0, h - 1, nrows).astype(np.int32)
    cols = np.linspace(0, w - 1, ncols).astype(np.int32)
    rows += np.random.normal(mean, std, size=rows.shape).astype(np.int32)
    rows += np.random.normal(mean, std, size=cols.shape).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:, 0], 0, h - 1, deformed[:, 0])
    np.clip(deformed[:, 1], 0, w - 1, deformed[:, 1])

    trans = PiecewiseAffineTransform()
    trans.estimate(anchors, deformed.astype(np.int32))
    warped = warp(mask, trans)
    warped *= mask
    blured = cv2.GaussianBlur(warped.astype(float), (5, 5), 3)
    return blured


def get_five_key(landmarks_68):
    # get the five key points by using the landmarks
    leye_center = (landmarks_68[36] + landmarks_68[39]) * 0.5
    reye_center = (landmarks_68[42] + landmarks_68[45]) * 0.5
    nose = landmarks_68[33]
    lmouth = landmarks_68[48]
    rmouth = landmarks_68[54]
    leye_left = landmarks_68[36]
    leye_right = landmarks_68[39]
    reye_left = landmarks_68[42]
    reye_right = landmarks_68[45]
    out = [tuple(x.astype('int32')) for x in [
        leye_center, reye_center, nose, lmouth, rmouth, leye_left, leye_right, reye_left, reye_right
    ]]
    return out


def remove_eyes(image, landmarks, opt):
    ##l: left eye; r: right eye, b: both eye
    if opt == 'l':
        (x1, y1), (x2, y2) = landmarks[5:7]
    elif opt == 'r':
        (x1, y1), (x2, y2) = landmarks[7:9]
    elif opt == 'b':
        (x1, y1), (x2, y2) = landmarks[:2]
    else:
        print('wrong region')
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    if opt != 'b':
        dilation *= 4
    line = binary_dilation(line, iterations=dilation)
    return line


def remove_nose(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    return line


def remove_mouth(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[3:5]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    return line


def blend_fake_to_real(realimg, real_lmk, fakeimg, fakemask, fake_lmk, deformed_fakemask, type, mag):
    # source: fake image
    # target: real image
    realimg = ((realimg + 1) / 2 * 255).astype(np.uint8)
    fakeimg = ((fakeimg + 1) / 2 * 255).astype(np.uint8)
    H, W, C = realimg.shape
    #由于我们已经做过对齐，这里可以直接用。原代码是做了对齐操作的. 这个src就是fake
    aligned_src = fakeimg
    src_mask = deformed_fakemask
    src_mask = src_mask > 0  # (H, W)

    tgt_mask = np.asarray(src_mask, dtype=np.uint8)
    tgt_mask = mask_postprocess(tgt_mask)

    ct_modes = ['rct-m', 'rct-fs', 'avg-align', 'faceswap']
    mode_idx = np.random.randint(len(ct_modes))
    mode = ct_modes[mode_idx]

    if mode != 'faceswap':
        c_mask = tgt_mask / 255.
        c_mask[c_mask > 0] = 1
        if len(c_mask.shape) < 3:
            c_mask = np.expand_dims(c_mask, 2)
        src_crop = color_transfer(mode, aligned_src, realimg, c_mask)
    else:
        c_mask = tgt_mask.copy()
        c_mask[c_mask > 0] = 255
        masked_tgt = faceswap.apply_mask(realimg, c_mask)
        masked_src = faceswap.apply_mask(aligned_src, c_mask)
        src_crop = faceswap.correct_colours(masked_tgt, masked_src, np.array(real_lmk))

    if tgt_mask.mean() < 0.005 or src_crop.max() == 0:
        out_blend = realimg
    else:
        if type == 0:
            out_blend, a_mask = alpha_blend_fea(src_crop, realimg, tgt_mask,
                                                featherAmount=0.2 * np.random.rand())
        elif type == 1:
            b_mask = (tgt_mask * 255).astype(np.uint8)
            l, t, w, h = cv2.boundingRect(b_mask)
            center = (int(l + w / 2), int(t + h / 2))
            out_blend = cv2.seamlessClone(src_crop, realimg, b_mask, center, cv2.NORMAL_CLONE)
        else:
            out_blend = copy_fake_to_real(realimg, src_crop, tgt_mask, mag)

    return out_blend, tgt_mask


def copy_fake_to_real(realimg, fakeimg, mask, mag):
    mask = np.expand_dims(mask, 2)
    newimg = fakeimg * mask * mag + realimg * (1 - mask) + realimg * mask * (1 - mag)
    return newimg


class synthesizer(nn.Module):
    def __init__(self,config):
        super(synthesizer, self).__init__()
        self.netG = TransferModel(config=config,num_region=10, num_type=4, num_mag=1, inc=6)
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transforms = T.Compose([T.ToTensor(), normalize])

    def parse(self, img, reg, real_lmk, fakemask):
        five_key = get_five_key(real_lmk)
        if reg == 0:
            mask = remove_eyes(img, five_key, 'l')
        elif reg == 1:
            mask = remove_eyes(img, five_key, 'r')
        elif reg == 2:
            mask = remove_eyes(img, five_key, 'b')
        elif reg == 3:
            mask = remove_nose(img, five_key)
        elif reg == 4:
            mask = remove_mouth(img, five_key)
        elif reg == 5:
            mask = remove_nose(img, five_key) + remove_eyes(img, five_key, 'l')
        elif reg == 6:
            mask = remove_nose(img, five_key) + remove_eyes(img, five_key, 'r')
        elif reg == 7:
            mask = remove_nose(img, five_key) + remove_eyes(img, five_key, 'b')
        elif reg == 8:
            mask = remove_nose(img, five_key) + remove_mouth(img, five_key)
        elif reg == 9:
            mask = remove_eyes(img, five_key, 'b') + remove_nose(img, five_key) + remove_mouth(img, five_key)
        else:
            mask = generate_random_mask(fakemask)
        mask = random_deform(mask, 5, 5)
        return mask * 1.0

    def get_variable(self, inputs, cuda=False, **kwargs):
        if type(inputs) in [list, np.ndarray]:
            inputs = torch.Tensor(inputs)
        if cuda:
            out = Variable(inputs.cuda(), **kwargs)
        else:
            out = Variable(inputs, **kwargs)
        return out

    def calculate(self, logits):
        if logits.shape[1] != 1:
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1, self.get_variable(action, requires_grad=False))
        else:
            probs = torch.sigmoid(logits)
            log_prob = torch.log(torch.sigmoid(logits))
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs
            selected_log_prob = log_prob
        return entropy, selected_log_prob[:, 0], action[:, 0]

    def forward(self, img, fake_img, real_lmk, fake_lmk, real_mask, fake_mask, label=None):
        # based on pair_dataset, here, img always is real, fake_img always is fake
        region_num, type_num, mag = self.netG(torch.cat((img, fake_img), 1))
        reg_etp, reg_log_prob, reg = self.calculate(region_num)
        type_etp, type_log_prob, type = self.calculate(type_num)
        mag_etp, mag_log_prob, mag = self.calculate(mag)
        entropy = reg_etp + type_etp + mag_etp
        log_prob = reg_log_prob + type_log_prob + mag_log_prob
        newlabel = []
        typelabel = []
        maglabel = []
        magmask = []
        #####################
        alt_img = torch.ones(img.shape)
        alt_mask = np.zeros((img.shape[0], 16, 16))
        if label is None:
            label=np.zeros(img.shape[0])
        for i in range(img.shape[0]):
            imgcp = np.transpose(img[i].cpu().numpy(), (1, 2, 0)).copy()
            fake_imgcp = np.transpose(fake_img[i].cpu().numpy(), (1, 2, 0)).copy()
            ##only work for real imgs and not do-nothing choice
            if label[i] == 0 and type[i] != 3:
                mask = self.parse(fake_imgcp, reg[i], fake_lmk[i].cpu().numpy(),
                                  fake_mask[i].cpu().numpy())
                newimg, newmask = blend_fake_to_real(imgcp, real_lmk[i].cpu().numpy(),
                                                     fake_imgcp, fake_mask.cpu().numpy(),
                                                     fake_lmk[i].cpu().numpy(), mask, type[i],
                                                     mag[i].detach().cpu().numpy())
                newimg = self.transforms(Image.fromarray(np.array(newimg, dtype=np.uint8)))
                newlabel.append(int(1))
                typelabel.append(int(type[i].cpu().numpy()))
                if type[i] == 2:
                    magmask.append(int(1))
                else:
                    magmask.append(int(0))
            else:
                newimg = self.transforms(Image.fromarray(np.array((imgcp + 1) / 2 * 255, dtype=np.uint8)))
                newmask =real_mask[i].squeeze(2)[:,:,0].cpu().numpy()
                newlabel.append(int(label[i]))
                if label[i] == 0:
                    typelabel.append(int(3))
                else:
                    typelabel.append(int(4))
                magmask.append(int(0))
            if newmask is None:
                newmask = np.zeros((16, 16))
            newmask = cv2.resize(newmask, (16, 16), interpolation=cv2.INTER_CUBIC)
            alt_img[i] = newimg
            alt_mask[i] = newmask

        alt_mask = torch.from_numpy(alt_mask.astype(np.float32)).unsqueeze(1)
        newlabel = torch.tensor(newlabel)
        typelabel = torch.tensor(typelabel)
        maglabel = mag
        magmask = torch.tensor(magmask)
        return log_prob, entropy, alt_img.detach(), alt_mask.detach(), \
            newlabel.detach(), typelabel.detach(), maglabel.detach(), magmask.detach()


if __name__ == '__main__':

    with open(r'H:\code\DeepfakeBench\training\config\detector\sladd_xception.yaml', 'r') as f:
        config = yaml.safe_load(f)
    syn=synthesizer(config=config).cuda()
    config['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config['sample_size']=256
    config['with_mask']=True
    config['with_landmark']=True
    config['use_data_augmentation']=True
    config['data_aug']['rotate_prob']=1
    train_set = pairDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        imgs,lmks,msks=batch['image'].cuda(),batch['landmark'].cuda(),batch['mask'].cuda()
        half = len(imgs) // 2
        img, fake_img, real_lmk, fake_lmk, real_mask, fake_mask = imgs[:half],imgs[half:],lmks[:half],lmks[half:],msks[:half],msks[half:]
        log_prob, entropy, new_img, alt_mask, label, type_label, mag_label, mag_mask = \
        syn(img, fake_img, real_lmk, fake_lmk, real_mask, fake_mask)

        if iteration > 10:
            break
    ...