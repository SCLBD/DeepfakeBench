from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

import torchvision.models as models

def add_gaussian_noise(ins, mean=0, stddev=0.1):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return ins + noise

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class UnflattenLayer(nn.Module):
    def __init__(self, width):
        super(UnflattenLayer, self).__init__()
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), -1, self.width, self.width)

class VAE_Encoder(nn.Module):
    ''' 
    VAE_Encoder: Encode image into std and logvar 
    '''

    def __init__(self, latent_dim=256):
        super(VAE_Encoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-1],
            FlattenLayer()
        )

        self.l_mu = nn.Linear(512, latent_dim)
        self.l_var = nn.Linear(512, latent_dim)

    def encode(self, x):
        hidden = self.resnet(x)
        mu = self.l_mu(hidden)
        logvar = self.l_var(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std

        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class VAE_Decoder(nn.Module):
    ''' 
    VAE_Decoder: Decode noise to image
    '''

    def __init__(self, latent_dim, output_dim=3):
        super(VAE_Decoder, self).__init__()        
        self.convs = nn.Sequential(
            UnflattenLayer(width=1),
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 384, 4, 2, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.convs(z)

class ImageAE(nn.Module):
    # VAE architecture
    def __init__(self):
        super(ImageAE, self).__init__()
        latent_dim = 512
        self.enc = VAE_Encoder(latent_dim)
        self.dec = VAE_Decoder(latent_dim)        

    def forward(self, x):
        z, *_ = self.enc(x)
        out = self.dec(z)
        
        return out
    
    def load_ckpt(self, enc_path,  dec_path):
        self.enc.load_state_dict(torch.load(enc_path, map_location='cpu'))
        self.dec.load_state_dict(torch.load(dec_path, map_location='cpu'))

    
def get_pretraiend_ae(enc_path='pretrained/ae/vae/enc.pth', dec_path='pretrained/ae/vae/dec1.pth'):
    ae = ImageAE()
    ae.load_ckpt(enc_path, dec_path)
    print('load image auto-encoder')
    ae.eval()
    return ae

# from networks.pix2pix_network import UnetGenerator
def get_pretraiend_unet(path='pretrained/ae/unet/ckpt_srm.pth'):
    unet = UnetGenerator(3, 3, 8)
    unet.load_state_dict(torch.load(path, map_location='cpu'))
    print('load Unet')
    unet.eval()
    return unet

if __name__ == "__main__":
    ae = get_pretraiend_ae()
    print(ae)
