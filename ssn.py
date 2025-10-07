import math

import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F

from blocks import Conv3x3, MaxPool2x2, DoubleConv
from lib import (CalcAssoc, CalcPixelFeats, CalcSpixelFeats, InitSpixelFeats, RelToAbsIndex, Smear)
from utils.utiltool import init_grid


class FeatureExtactor(nn.Module):
    def __init__(self, n_filters=64, in_ch=5, out_ch=20):
        super().__init__()
        self.conv1 = DoubleConv(in_ch, n_filters)
        self.pool1 = MaxPool2x2()
        self.conv2 = DoubleConv(n_filters, n_filters)
        self.pool2 = MaxPool2x2()
        self.conv3 = DoubleConv(n_filters, n_filters)
        self.conv4 = Conv3x3(3*n_filters+in_ch, out_ch-in_ch, act=True)
    
    def forward(self, x):
        f1 = self.conv1(x)
        p1 = self.pool1(f1)
        f2 = self.conv2(p1)
        p2 = self.pool2(f2)
        f3 = self.conv3(p2)

        # Resize feature maps
        f2_rsz = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3_rsz = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate multi-level features and fuse them
        f_cat = torch.cat([x, f1, f2_rsz, f3_rsz], dim=1)
        f_out = self.conv4(f_cat)

        y = torch.cat([x, f_out], dim=1)

        return y

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            # nn.Conv2d(dim*len(time_steps), dim, 1),
            # # if len(time_steps)>1
            # # else None,
            # nn.ReLU(inplace=True),
            # if len(time_steps)>1
            # else None,
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SSN(nn.Module):
    def __init__(
        self,
        n_iters,
        n_spixels,
        inner_channel=None,
        channel_multiplier=None
    ):
        super().__init__()

        self.n_spixels = n_spixels
        self.n_iters = n_iters
        self._cached = False
        self._ops = {}
        self._layout = (None, 1, 1)
        self.out_channel = 8
        self.decoder=nn.Sequential(
            nn.Conv2d(in_channels=inner_channel*channel_multiplier[0], out_channels=self.out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, featlist):
        featall = self.decoder(featlist[0])


        if self.training:
            # Training mode
            # Use cached objects
            ops, (_, nw_spixels, nh_spixels) = self.get_ops_and_layout(featall , ofa=True)
        else:
            # Evaluation mode
            # Every time update the objects
            ops, (_, nw_spixels, nh_spixels) = self.get_ops_and_layout(featall , ofa=False)
        # Forward

        spf = ops['init_spixels'](featall.detach())

        # Iterations
        for itr in range(self.n_iters):
            Q = self.nd2Q(ops['calc_neg_dist'](featall , spf))
            spf = ops['map_p2sp'](featall , Q )


        return Q, ops, featlist, spf, featall

    @staticmethod
    def nd2Q(neg_dist):
        # Use softmax to compute pixel-superpixel relative soft-associations (degree of membership)
        return F.softmax(neg_dist, dim=1)

    def get_ops_and_layout(self, x, ofa=False):#生成超像素
        if ofa and self._cached:
            return self._ops, self._layout
        
        b, _, h, w = x.size()   # Get size of the input

        # Initialize grid
        init_idx_map, n_spixels, nw_spixels, nh_spixels = init_grid(self.n_spixels, w, h)
        init_idx_map = torch.IntTensor(init_idx_map).expand(b, 1, h, w).to(x.device)

        # Contruct operation modules
        init_spixels = InitSpixelFeats(n_spixels, init_idx_map)
        map_p2sp = CalcSpixelFeats(nw_spixels, nh_spixels, init_idx_map)
        map_sp2p = CalcPixelFeats(nw_spixels, nh_spixels, init_idx_map)
        calc_neg_dist = CalcAssoc(nw_spixels, nh_spixels, init_idx_map)
        map_idx = RelToAbsIndex(nw_spixels, nh_spixels, init_idx_map)
        smear = Smear(n_spixels)

        ops = {
            'init_spixels': init_spixels,
            'map_p2sp': map_p2sp,
            'map_sp2p': map_sp2p,
            'calc_neg_dist': calc_neg_dist,
            'map_idx': map_idx,
            'smear': smear
        }

        if ofa:
            self._ops = ops
            self._layout = (init_idx_map, nw_spixels, nh_spixels)
            self._cached = True

        return ops, (init_idx_map, nw_spixels, nh_spixels)