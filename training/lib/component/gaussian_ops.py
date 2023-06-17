import cv2
import numpy as np
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma=0.1, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.kernel_size = kernel_size
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
                    dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        if self.training:
            return self.conv(input, weight=self.weight, groups=self.groups, padding=self.kernel_size//2)
        else:
            return input

class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.1, clip=1):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std
        self.clip = clip

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(self.mean, self.std)
            return torch.clamp(x + noise, -self.clip, self.clip)
        else:
            return x


if __name__ == "__main__":
    im = cv2.imread('E:\SRM\component\FF-F2F_0.png')
    im_ten = im/255*2-1
    im_ten = torch.from_numpy(im_ten).unsqueeze(0).permute(0, 3, 1, 2).float()
    blur = GaussianSmoothing(channels=3, kernel_size=7, sigma=0.8)
    noise = GaussianNoise()

    noise_im = torch.clamp(noise(im_ten), -1, 1)
    blur_im = blur(im_ten)
    print(blur_im.size())

    def t2im(t):

        t = (t+1)/2*255
        im = t.squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        return im

    cv2.imshow('ori', im)
    cv2.imshow('blur', t2im(blur_im))
    cv2.imshow('noise', t2im(noise_im))

    cv2.waitKey()
