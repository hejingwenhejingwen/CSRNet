import functools
import torch.nn as nn
import torch
import models.archs.arch_util as arch_util


class BaseNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=32):
        super(BaseNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 2, 1, bias=True)

        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        self.upconv = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv, self.HRconv, self.conv_last], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv_first(identity)
        out = self.recon_trunk(out)
        out = self.act(self.pixel_shuffle(self.upconv(out)))
        out = self.conv_last(self.act(self.HRconv(out)))
        return identity + out 


# #################
# CResMD
class CResMDNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=32, cond_dim=2):
        super(CResMDNet, self).__init__()

        # condition mapping
        self.global_scale = nn.Linear(cond_dim, out_nc, bias=True)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 2, 1, bias=True)

        basic_block = functools.partial(ResidualBlock_CRes, nf=nf, cond_dim=cond_dim)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        self.upconv = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv, self.HRconv, self.conv_last], 0.1)
        arch_util.initialize_weights([self.global_scale], 0.1)

    def forward(self, x):
        identity = x[0]
        cond = x[1]

        global_scale = self.global_scale(cond)

        out = self.conv_first(identity)
        out, _ = self.recon_trunk((out, cond))
        out = self.act(self.pixel_shuffle(self.upconv(out)))
        out = self.conv_last(self.act(self.HRconv(out)))
        return identity + out * global_scale.view(-1, 3, 1, 1)


class ResidualBlock_CRes(nn.Module):
    '''Residual block with controllable residual connections
    ---Conv-ReLU-Conv-x-+-
     |________________|
    '''

    def __init__(self, nf=64, cond_dim=2):
        super(ResidualBlock_CRes, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

        self.local_scale = nn.Linear(cond_dim, nf, bias=True)

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)
        arch_util.initialize_weights([self.local_scale], 0.1)

    def forward(self, x):
        identity = x[0]
        cond = x[1]

        local_scale = self.local_scale(cond)
        out = self.conv1(identity)
        out = self.conv2(self.act(out))
        return identity + out * local_scale.view(-1, 64, 1, 1), cond