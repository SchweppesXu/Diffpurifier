# Kaiyu Li
# https://github.com/likyoo
#

import torch.nn as nn
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from blocks import (Conv3x3, MaxPool2x2, ResBlock, ResBlock2, DecBlock)
from ssn import SSN
from .base_model import BaseModel

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # input1, pad_left1, pad_right1, pad_top1, pad_bottom1 = pad_tensor(x, 2)
        x = self.conv1(x)
        # x = pad_tensor_back(x, pad_left1, pad_right1, pad_top1, pad_bottom1)
        identity = x

        x = self.bn1(x)
        x = self.activation(x)
        # input2, pad_left2, pad_right2, pad_top2, pad_bottom2 = pad_tensor(x, 2)
        x = self.conv2(x)
        # x = pad_tensor_back(x, pad_left2, pad_right2, pad_top2, pad_bottom2)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class channel_reduce_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(channel_reduce_block, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        # self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.activation(output)
        return output



class RefineNet(nn.Module):
    def __init__(self, in_ch, out_ch):#16,2
        super().__init__()
        # C = [in_ch, 32, 64, 128]   # output4,3,2,1
        C = [in_ch, in_ch, in_ch, in_ch]
        # Fusion layers
        self.fuse1 = Conv3x3(in_ch+C[0], in_ch, bn=True, act=True)#32,16
        self.fuse2 = Conv3x3(in_ch+C[1], in_ch, bn=True, act=True)#48,16
        self.fuse3 = Conv3x3(in_ch+C[2], in_ch, bn=True, act=True)#80,16
        self.fuse4 = Conv3x3(in_ch+C[3], in_ch, bn=True, act=True)#144,16

        self.conv_out = nn.Sequential(
            Conv3x3(in_ch, in_ch, bn=True, act=True),
            Conv3x3(in_ch, out_ch, bn=False, act=False)
        )

    def forward(self, x, feats_to_fuse):
        y = x + self.fuse1(torch.cat([x, feats_to_fuse[0]], dim=1))
        interp_configs = dict(size=x.shape[2:], mode='bilinear', align_corners=True)#都缩放到256×256
        y = y + self.fuse2(torch.cat([x, F.interpolate(feats_to_fuse[1], **interp_configs)], dim=1))
        y = y + self.fuse3(torch.cat([x, F.interpolate(feats_to_fuse[2], **interp_configs)], dim=1))
        y = y + self.fuse4(torch.cat([x, F.interpolate(feats_to_fuse[3], **interp_configs)], dim=1))

        return self.conv_out(y)

class RefineNet_ddpm_escamafter(nn.Module):
    def __init__(self, in_ch, out_ch):#16,2
        super().__init__()
        # C = [in_ch, 32, 64, 128]   # output8,3,2,1
        C = [in_ch, in_ch, in_ch, in_ch]
        # Fusion layers
        self.ca = ChannelAttention( 8 * 2, ratio= 8)
        self.ca1 = ChannelAttention( 8, ratio=8 // 2)

        self.fuse1 = Conv3x3(in_ch+C[0], in_ch, bn=True, act=True)#32,16
        self.conv_out = Conv3x3(in_ch+C[0], out_ch, bn=False, act=False)

    def forward(self, x, feats_to_fuse):
        # y = x + self.fuse1(torch.cat([x, feats_to_fuse], dim=1))
        out = torch.cat([x, feats_to_fuse], 1)
        intra = torch.sum(torch.stack((x, feats_to_fuse)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 2, 1, 1))
        # y = self.fuse1(torch.cat([x, feats_to_fuse], dim=1))
        return self.conv_out(out)

class RefineNet_ddpm_escambefore(nn.Module):
    def __init__(self, in_ch, out_ch):#16,2
        super().__init__()
        # C = [in_ch, 32, 64, 128]   # output4,3,2,1
        C = [in_ch, in_ch, in_ch, in_ch]
        # Fusion layers
        self.ca = ChannelAttention(20 * 5, ratio=20)
        self.ca1 = ChannelAttention(20, ratio=20 // 5)

        self.fuse1 = Conv3x3(in_ch+C[0], in_ch, bn=True, act=True)#32,16
        self.conv_out = Conv3x3(in_ch+C[0]*4, out_ch, bn=False, act=False)

    def forward(self, x, feats_to_fuse):
        # y = x + self.fuse1(torch.cat([x, feats_to_fuse], dim=1))
        out = torch.cat([x, feats_to_fuse[0],feats_to_fuse[1],feats_to_fuse[2],feats_to_fuse[3]], 1)
        intra = torch.sum(torch.stack((x, feats_to_fuse[0],feats_to_fuse[1],feats_to_fuse[2],feats_to_fuse[3])), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 5, 1, 1))
        # y = self.fuse1(torch.cat([x, feats_to_fuse], dim=1))
        return self.conv_out(out)

class RefineNet_ablation(nn.Module):
    def __init__(self, in_ch):  # 16,2
        super().__init__()
        self.fuse1 = Conv3x3(in_ch, in_ch, bn=True, act=True)#32,16
        self.conv_out = Conv3x3(in_ch, 2, bn=False, act=False)

    def forward(self,feats_to_fuse):

        y = self.fuse1(feats_to_fuse)
        return self.conv_out(feats_to_fuse)


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
             self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2,)

    def forward(self, x):
        x = self.up(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class SNUNet_ECAM(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, out_ch=2):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        self.filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, self.filters[0], self.filters[0])
        self.conv1_0 = conv_block_nested(self.filters[0], self.filters[1], self.filters[1])
        self.Up1_0 = up(self.filters[1])
        self.conv2_0 = conv_block_nested(self.filters[1], self.filters[2], self.filters[2])
        self.Up2_0 = up(self.filters[2])
        self.conv3_0 = conv_block_nested(self.filters[2], self.filters[3], self.filters[3])
        self.Up3_0 = up(self.filters[3])
        self.conv4_0 = conv_block_nested(self.filters[3], self.filters[4], self.filters[4])
        self.Up4_0 = up(self.filters[4])

        self.conv0_1 = conv_block_nested(self.filters[0] * 2 + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_1 = conv_block_nested(self.filters[1] * 2 + self.filters[2], self.filters[1], self.filters[1])
        self.Up1_1 = up(self.filters[1])
        self.conv2_1 = conv_block_nested(self.filters[2] * 2 + self.filters[3], self.filters[2], self.filters[2])
        self.Up2_1 = up(self.filters[2])
        self.conv3_1 = conv_block_nested(self.filters[3] * 2 + self.filters[4], self.filters[3], self.filters[3])
        self.Up3_1 = up(self.filters[3])

        self.conv0_2 = conv_block_nested(self.filters[0] * 3 + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_2 = conv_block_nested(self.filters[1] * 3 + self.filters[2], self.filters[1], self.filters[1])
        self.Up1_2 = up(self.filters[1])
        self.conv2_2 = conv_block_nested(self.filters[2] * 3 + self.filters[3], self.filters[2], self.filters[2])
        self.Up2_2 = up(self.filters[2])

        self.conv0_3 = conv_block_nested(self.filters[0] * 4 + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_3 = conv_block_nested(self.filters[1] * 4 + self.filters[2], self.filters[1], self.filters[1])
        self.Up1_3 = up(self.filters[1])

        self.conv0_4 = conv_block_nested(self.filters[0] * 5 + self.filters[1], self.filters[0], self.filters[0])

        self.ca = ChannelAttention(self.filters[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(self.filters[0], ratio=16 // 4)

        self.conv_final = nn.Conv2d(self.filters[0] * 4, out_ch, kernel_size=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)

        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.conv_final(out)

        return out


class Siam_NestedUNet_Conc(nn.Module):
    # SNUNet-CD without Attention
    def __init__(self, in_ch=3, out_ch=2):
        super(Siam_NestedUNet_Conc, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        self.filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 8]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, self.filters[0], self.filters[0])
        self.conv1_0 = conv_block_nested(self.filters[0], self.filters[1], self.filters[1])
        self.Up1_0 = up(self.filters[1])
        self.conv2_0 = conv_block_nested(self.filters[1], self.filters[2], self.filters[2])
        self.Up2_0 = up(self.filters[2])
        self.conv3_0 = conv_block_nested(self.filters[2], self.filters[3], self.filters[3])
        self.Up3_0 = up(self.filters[3])
        self.conv4_0 = conv_block_nested(self.filters[3], self.filters[4], self.filters[4])
        self.Up4_0 = up(self.filters[4])

        self.conv0_1 = conv_block_nested(self.filters[0] * 2 + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_1 = conv_block_nested(self.filters[1] * 2 + self.filters[2], self.filters[1], self.filters[1])
        self.Up1_1 = up(self.filters[1])
        self.conv2_1 = conv_block_nested(self.filters[2] * 2 + self.filters[3], self.filters[2], self.filters[2])
        self.Up2_1 = up(self.filters[2])
        self.conv3_1 = conv_block_nested(self.filters[3] * 2 + self.filters[4], self.filters[3], self.filters[3])
        self.Up3_1 = up(self.filters[3])

        self.conv0_2 = conv_block_nested(self.filters[0] * 3 + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_2 = conv_block_nested(self.filters[1] * 3 + self.filters[2], self.filters[1], self.filters[1])
        self.Up1_2 = up(self.filters[1])
        self.conv2_2 = conv_block_nested(self.filters[2] * 3 + self.filters[3], self.filters[2], self.filters[2])
        self.Up2_2 = up(self.filters[2])

        self.conv0_3 = conv_block_nested(self.filters[0] * 4 + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_3 = conv_block_nested(self.filters[1] * 4 + self.filters[2], self.filters[1], self.filters[1])
        self.Up1_3 = up(self.filters[1])

        self.conv0_4 = conv_block_nested(self.filters[0] * 5 + self.filters[1], self.filters[0], self.filters[0])

        self.final1 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.final2 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.final3 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.final4 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.conv_final = nn.Conv2d(out_ch * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))


        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        output = self.conv_final(torch.cat([output1, output2, output3, output4], 1))
        return (output1, output2, output3, output4)



def PCA_Batch_Feat(X, k, center=True):
    """
    param X: BxCxHxW
    param k: scalar
    return:
    """
    B, C, H, W = X.shape
    X = X.permute(0, 2, 3, 1)  # BxHxWxC
    X = X.reshape(B, H * W, C)
    U,S,V= torch.pca_lowrank(X, q=k)
    Y=torch.matmul(X,V)
    Y = Y.reshape(B, H, W, k)  # BxHxWxk
    Y = Y.permute(0, 3, 1, 2)  # B, k, H, W
    return Y

class SNUNet_Conc(nn.Module):
    # SNUNet-CD without Attention
    def __init__(self, out_ch=2):
        super(SNUNet_Conc, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        self.filters0 = [n1*8, n1 * 8, n1 * 16, n1 * 16, n1 * 32]
        self.filters = [n1 , n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # self.filters = [n1*8, n1 * 8, n1 * 16, n1 * 16, n1 * 32]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up1_0 = up(self.filters[1])
        self.Up2_0 = up(self.filters[2])
        self.Up3_0 = up(self.filters[3])
        self.Up4_0 = up(self.filters[4])

        # self.conv2d_1 = nn.Conv2d(in_channels=self.filters0[0], out_channels=self.filters[0], kernel_size=3, stride=1, padding=1)
        # self.conv2d_2 = nn.Conv2d(in_channels=self.filters0[1], out_channels=self.filters[1], kernel_size=3, stride=1, padding=1)
        # self.conv2d_3 = nn.Conv2d(in_channels=self.filters0[2], out_channels=self.filters[2], kernel_size=3, stride=1, padding=1)
        # self.conv2d_4 = nn.Conv2d(in_channels=self.filters0[3], out_channels=self.filters[3], kernel_size=3, stride=1, padding=1)
        # self.conv2d_5 = nn.Conv2d(in_channels=self.filters0[4], out_channels=self.filters[4], kernel_size=3, stride=1, padding=1)

        self.conv2d_1 = channel_reduce_block(self.filters0[0], self.filters[0])
        self.conv2d_2 = channel_reduce_block(self.filters0[1], self.filters[1])
        self.conv2d_3 = channel_reduce_block(self.filters0[2], self.filters[2])
        self.conv2d_4 = channel_reduce_block(self.filters0[3], self.filters[3])
        self.conv2d_5 = channel_reduce_block(self.filters0[4], self.filters[4])

        self.conv0_1 = conv_block_nested(self.filters[0] * 2 + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_1 = conv_block_nested(self.filters[1] * 2 + self.filters[2], self.filters[1], self.filters[1])
        self.Up1_1 = up(self.filters[1])
        self.conv2_1 = conv_block_nested(self.filters[2] * 2 + self.filters[3], self.filters[2], self.filters[2])
        self.Up2_1 = up(self.filters[2])
        self.conv3_1 = conv_block_nested(self.filters[3] * 2 + self.filters[4], self.filters[3], self.filters[3])
        self.Up3_1 = up(self.filters[3])

        self.conv0_2 = conv_block_nested(self.filters[0] * 3 + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_2 = conv_block_nested(self.filters[1] * 3 + self.filters[2], self.filters[1], self.filters[1])
        self.Up1_2 = up(self.filters[1])
        self.conv2_2 = conv_block_nested(self.filters[2] * 3 + self.filters[3], self.filters[2], self.filters[2])
        self.Up2_2 = up(self.filters[2])

        self.conv0_3 = conv_block_nested(self.filters[0] * 4 + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_3 = conv_block_nested(self.filters[1] * 4 + self.filters[2], self.filters[1], self.filters[1])
        self.Up1_3 = up(self.filters[1])

        self.conv0_4 = conv_block_nested(self.filters[0] * 5 + self.filters[1], self.filters[0], self.filters[0])

        self.final1 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.final2 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.final3 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.final4 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.conv_final = nn.Conv2d(out_ch * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, featA,featB):
        # for i in range(1, len(self.time_steps)):
        #     x0_0A = torch.cat((x0_0A, featA[i][self.feat_scales[0]]), dim=1)
        #     x0_0B = torch.cat((x0_0B, featB[i][self.feat_scales[0]]), dim=1)
        #     # fAfB是同一噪声等级不同时间步的特征concat
        # print(featA[0].shape)
        x0_0A =  self.conv2d_1(featA[0])
        x0_0B =  self.conv2d_1(featB[0])


        # x0_0A =  featA[0]
        # x0_0B =  featB[0]
        # print(featA[1].shape)
        x1_0A =  self.conv2d_2(featA[1])
        x1_0B =  self.conv2d_2(featB[1])
        # x1_0A = PCA_Batch_Feat(x1_0A, self.filters[1])
        # x1_0B = PCA_Batch_Feat(x1_0B, self.filters[1])
        # x1_0A = featA[1]
        # x1_0B = featB[1]
        # print(featA[2].shape)
        x2_0A =  self.conv2d_3(featA[2])
        x2_0B =  self.conv2d_3(featB[2])
        # x2_0A = PCA_Batch_Feat(x2_0A, self.filters[2])
        # x2_0B = PCA_Batch_Feat(x2_0B, self.filters[2])
        # x2_0A = featA[2]
        # x2_0B = featB[2]

        x3_0A =  self.conv2d_4(featA[3])
        x3_0B =  self.conv2d_4(featB[3])
        # x3_0A = PCA_Batch_Feat(x3_0A, self.filters[3])
        # x3_0B = PCA_Batch_Feat(x3_0B, self.filters[3])
        # x3_0A = featA[3]
        # x3_0B = featB[3]

        x4_0B =  self.conv2d_5(featB[4])
        # x4_0B = PCA_Batch_Feat(x4_0B, self.filters[4])

        # x4_0B = featB[4]

        del featA,featB
        torch.cuda.empty_cache()


        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        output = self.conv_final(torch.cat([output1, output2, output3, output4], 1))


        return (output1, output2, output3, output4, output)

class Attention_Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, pool_window=6, add_input=False):
        super(Attention_Embedding, self).__init__()
        self.add_input = add_input
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.SE = nn.Sequential(
           # nn.AvgPool2d(kernel_size=pool_window + 1, stride=1, padding=pool_window // 2),
            nn.Conv2d(in_channels*2, in_channels*2 // reduction, 1),
            nn.BatchNorm2d(in_channels*2 // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels*2 // reduction, out_channels, 1),
            nn.Sigmoid())

    def forward(self, high_feat, low_feat):
        b, c, h, w = low_feat.size()
        avg_out = self.avg_pool(high_feat)
        max_out = self.max_pool(high_feat)
        high_feat = torch.cat([avg_out, max_out], dim=1)
        A = self.SE(high_feat)
        A = F.upsample(A, (h, w), mode='bilinear')

        output = low_feat * A
        if self.add_input:
            output += low_feat

        return output

class SNUNet_ConcFEM(nn.Module):
    def __init__(self, out_ch=2):
        super(SNUNet_ConcFEM, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        self.filters0 = [n1*8, n1 * 8, n1 * 16, n1 * 16, n1 * 32]
        self.filters = [n1 , n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up1_0 = up(self.filters[1])
        self.Up2_0 = up(self.filters[2])
        self.Up3_0 = up(self.filters[3])
        self.Up4_0 = up(self.filters[4])

        self.conv2d_1 = channel_reduce_block(self.filters0[0], self.filters[0])
        self.conv2d_2 = channel_reduce_block(self.filters0[1], self.filters[1])
        self.conv2d_3 = channel_reduce_block(self.filters0[2], self.filters[2])
        self.conv2d_4 = channel_reduce_block(self.filters0[3], self.filters[3])
        self.conv2d_5 = channel_reduce_block(self.filters0[4], self.filters[4])

        self.conv0_1 = conv_block_nested(self.filters[0] * 2 + self.filters[1], self.filters[0], self.filters[0])
        self.AE0_1 = Attention_Embedding(self.filters[1] * 2, self.filters[0]*2)
        self.conv1_1 = conv_block_nested(self.filters[1] * 2 + self.filters[2], self.filters[1], self.filters[1])
        self.AE1_1 = Attention_Embedding(self.filters[2] * 2, self.filters[1] * 2)
        self.Up1_1 = up(self.filters[1])
        self.conv2_1 = conv_block_nested(self.filters[2] * 2 + self.filters[3], self.filters[2], self.filters[2])
        self.AE2_1 = Attention_Embedding(self.filters[3] * 2, self.filters[2] * 2)
        self.Up2_1 = up(self.filters[2])
        self.conv3_1 = conv_block_nested(self.filters[3] * 2 + self.filters[4], self.filters[3], self.filters[3])
        self.AE3_1 = Attention_Embedding(self.filters[4] * 2, self.filters[3] * 2)
        self.Up3_1 = up(self.filters[3])

        self.conv0_2 = conv_block_nested(self.filters[0] * 3 + self.filters[1], self.filters[0], self.filters[0])
        self.AE0_2 = Attention_Embedding(self.filters[1], self.filters[0] * 3 )
        self.conv1_2 = conv_block_nested(self.filters[1] * 3 + self.filters[2], self.filters[1], self.filters[1])
        self.AE1_2 = Attention_Embedding(self.filters[2], self.filters[1] * 3)
        self.Up1_2 = up(self.filters[1])
        self.conv2_2 = conv_block_nested(self.filters[2] * 3 + self.filters[3], self.filters[2], self.filters[2])
        self.AE2_2 = Attention_Embedding(self.filters[3], self.filters[2] * 3)
        self.Up2_2 = up(self.filters[2])

        self.conv0_3 = conv_block_nested(self.filters[0] * 4 + self.filters[1], self.filters[0], self.filters[0])
        self.AE0_3 = Attention_Embedding(self.filters[1], self.filters[0] * 4)
        self.conv1_3 = conv_block_nested(self.filters[1] * 4 + self.filters[2], self.filters[1], self.filters[1])
        self.AE1_3 = Attention_Embedding(self.filters[2], self.filters[1] * 4 )
        self.Up1_3 = up(self.filters[1])

        self.conv0_4 = conv_block_nested(self.filters[0] * 5 + self.filters[1], self.filters[0], self.filters[0])
        self.AE0_4 = Attention_Embedding(self.filters[1], self.filters[0] * 5)

        self.final1 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.final2 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.final3 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.final4 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)
        self.conv_final = nn.Conv2d(out_ch * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, featA,featB):

        x0_0A =  self.conv2d_1(featA[0])
        x0_0B =  self.conv2d_1(featB[0])

        x1_0A =  self.conv2d_2(featA[1])
        x1_0B =  self.conv2d_2(featB[1])

        x2_0A =  self.conv2d_3(featA[2])
        x2_0B =  self.conv2d_3(featB[2])

        x3_0A =  self.conv2d_4(featA[3])
        x3_0B =  self.conv2d_4(featB[3])

        x4_0A = self.conv2d_5(featA[4])
        x4_0B =  self.conv2d_5(featB[4])

        del featA,featB
        torch.cuda.empty_cache()


        x0_1 = self.AE0_1(torch.cat([x1_0A, x1_0B], 1), torch.cat([x0_0A, x0_0B], 1))
        x0_1 = self.conv0_1(torch.cat([x0_1, self.Up1_0(x1_0B)], 1))

        x1_1 = self.AE1_1(torch.cat([x2_0A, x2_0B], 1), torch.cat([x1_0A, x1_0B], 1))
        x1_1 = self.conv1_1(torch.cat([x1_1,self.Up2_0(x2_0B)], 1))

        x0_2 = self.AE0_2(x1_1, torch.cat([x0_0A, x0_0B, x0_1], 1))
        x0_2 = self.conv0_2(torch.cat([x0_2,self.Up1_1(x1_1)], 1))

        x2_1 = self.AE2_1(torch.cat([x3_0A, x3_0B], 1), torch.cat([x2_0A, x2_0B], 1))
        x2_1 = self.conv2_1(torch.cat([x2_1,self.Up3_0(x3_0B)], 1))

        x1_2 = self.AE1_2(x2_1, torch.cat([x1_0A, x1_0B, x1_1], 1))
        x1_2 = self.conv1_2(torch.cat([x1_2, self.Up2_1(x2_1)], 1))

        x0_3 = self.AE0_3(x1_2, torch.cat([x0_0A, x0_0B, x0_1, x0_2], 1))
        x0_3 = self.conv0_3(torch.cat([x0_3, self.Up1_2(x1_2)], 1))

        x3_1 = self.AE3_1(torch.cat([x4_0A, x4_0B], 1), torch.cat([x3_0A, x3_0B], 1))
        x3_1 = self.conv3_1(torch.cat([x3_1, self.Up4_0(x4_0B)], 1))
        x2_2 = self.AE2_2(x3_1, torch.cat([x2_0A, x2_0B, x2_1], 1))
        x2_2 = self.conv2_2(torch.cat([x2_2, self.Up3_1(x3_1)], 1))
        x1_3 = self.AE1_3(x2_2, torch.cat([x1_0A, x1_0B, x1_1, x1_2], 1))
        x1_3 = self.conv1_3(torch.cat([x1_3, self.Up2_2(x2_2)], 1))
        x0_4 = self.AE0_4(x1_3, torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3], 1))
        x0_4 = self.conv0_4(torch.cat([x0_4, self.Up1_3(x1_3)], 1))


        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        output = self.conv_final(torch.cat([output1, output2, output3, output4], 1))


        return (output1, output2, output3, output4, output)

class SESNUNet(nn.Module):
    def __init__(
            self,
            n_iters,
            n_spixels,
            out_channels,
            inner_channel,
            channel_multiplier
    ):
        super().__init__()
        self.ssn = SSN(n_iters,n_spixels,inner_channel,channel_multiplier)
        self.cd_net = SNUNet_Conc(out_channels)
        self.conv_ds = nn.Sequential(
            Conv3x3(out_channels, 2, bn=False, act=False),
            nn.LogSoftmax(dim=1)
        )
        self.fuse_net = RefineNet_ddpm_escamafter(out_channels, 2)
        # self.act_out = nn.LogSoftmax(dim=1)
        self.omega2 = (0.01 * n_spixels) ** 2

    def forward(self, f1, f2, merge=True):
        # Compute Qs
        Q1, ops1, f1, spf1, pf1 = self.ssn(f1)
        Q2, ops2, f2, spf2, pf2 = self.ssn(f2)

        # with autocast():

        Q1_d, Q2_d = Q1.detach(), Q2.detach()

        # Extract pixel-level features
        # pf means pixel features and hf means hidden-layer features
        hf = self.cd_net(f1, f2)
        pf = hf[4]

            # Super-pixelation
        if merge:
            # Adaptive superpixel merging    spf是超像素的平均特征1，20，256
            b, c, s = spf1.size()
            spf1.detach_()
            rels = spf1.unsqueeze(-2) - spf1.unsqueeze(-1)
            rels = torch.exp(-(rels ** 2).sum(dim=1, keepdim=True) / self.omega2)
            # Filter out too small and invalid (>1) values to avoid possible noise
            rels[rels < 0.1] = 0
            rels[rels > 1.0] = 1
            coeffs = ops1['map_sp2p'](rels.view(b, s, s), Q1_d).view(b, 1, s, -1)
            spf1 = (coeffs * pf.view(b, c, 1, -1)).sum(-1) / (coeffs.sum(-1) + 1e-32)

            spf2.detach_()
            rels = spf2.unsqueeze(-2) - spf2.unsqueeze(-1)
            rels = torch.exp(-(rels ** 2).sum(dim=1, keepdim=True) / self.omega2)
            rels[rels < 0.1] = 0
            rels[rels > 1.0] = 1
            coeffs = ops2['map_sp2p'](rels.view(b, s, s), Q2_d).view(b, 1, s, -1)
            spf2 = (coeffs * pf.view(b, c, 1, -1)).sum(-1) / (coeffs.sum(-1) + 1e-32)
            del rels, coeffs
        else:
            spf1 = ops1['map_p2sp'](pf, Q1_d)
            spf2 = ops2['map_p2sp'](pf, Q2_d)

        pf1 = ops1['map_sp2p'](spf1, Q1_d)
        pf2 = ops2['map_sp2p'](spf2, Q2_d)

        # with autocast():
        pf_sp = pf1 + pf2
        prob_ds = self.conv_ds(pf_sp)

        # Pixel-level refinement
        pf_out = self.fuse_net(pf_sp, pf)

        # prob = self.act_out(pf_out)

        return pf_out, prob_ds, (Q1, Q2), (f1, f2)
        # return pf_out, prob_ds,(Q1, Q2),(ops1, ops2), (f1, f2)

class SESNUNet_ablation(nn.Module):
    def __init__(
            self,
            n_iters,
            n_spixels,
            out_channels,
            inner_channel,
            channel_multiplier
    ):
        super().__init__()

        self.ssn = SSN(n_iters, n_spixels, inner_channel, channel_multiplier)
        self.cd_net = SNUNet_Conc(out_channels)
        self.conv_ds = nn.Sequential(
            Conv3x3(out_channels, 2, bn=False, act=False),
            nn.LogSoftmax(dim=1)
        )
        self.fuse_net = RefineNet_ablation(out_channels)
        # self.act_out = nn.LogSoftmax(dim=1)
        self.omega2 = (0.01 * n_spixels) ** 2

    def forward(self, f1, f2, merge=True):

        # Compute Qs
        Q1, ops1, f1, spf1, pf1 = self.ssn(f1)
        Q2, ops2, f2, spf2, pf2 = self.ssn(f2)

        # with autocast():

        Q1_d, Q2_d = Q1.detach(), Q2.detach()

        # Extract pixel-level features
        # pf means pixel features and hf means hidden-layer features
        hf = self.cd_net(f1, f2)
        pf = hf[4]

        # Super-pixelation
        if merge:
            # Adaptive superpixel merging    spf是超像素的平均特征1，20，256
            b, c, s = spf1.size()
            spf1.detach_()
            rels = spf1.unsqueeze(-2) - spf1.unsqueeze(-1)
            rels = torch.exp(-(rels ** 2).sum(dim=1, keepdim=True) / self.omega2)
            # Filter out too small and invalid (>1) values to avoid possible noise
            rels[rels < 0.1] = 0
            rels[rels > 1.0] = 1
            coeffs = ops1['map_sp2p'](rels.view(b, s, s), Q1_d).view(b, 1, s, -1)
            spf1 = (coeffs * pf.view(b, c, 1, -1)).sum(-1) / (coeffs.sum(-1) + 1e-32)

            spf2.detach_()
            rels = spf2.unsqueeze(-2) - spf2.unsqueeze(-1)
            rels = torch.exp(-(rels ** 2).sum(dim=1, keepdim=True) / self.omega2)
            rels[rels < 0.1] = 0
            rels[rels > 1.0] = 1
            coeffs = ops2['map_sp2p'](rels.view(b, s, s), Q2_d).view(b, 1, s, -1)
            spf2 = (coeffs * pf.view(b, c, 1, -1)).sum(-1) / (coeffs.sum(-1) + 1e-32)
            del rels, coeffs
        else:
            spf1 = ops1['map_p2sp'](pf, Q1_d)
            spf2 = ops2['map_p2sp'](pf, Q2_d)

        pf1 = ops1['map_sp2p'](spf1, Q1_d)
        pf2 = ops2['map_sp2p'](spf2, Q2_d)

        # with autocast():
        pf_sp = pf1 + pf2
        prob_ds = self.conv_ds(pf_sp)

        # Pixel-level refinement
        # pf_out = self.fuse_net(pf)
        pf_out = pf
        # prob = self.act_out(pf_out)

        return pf_out, prob_ds, (Q1, Q2), (f1, f2)
        # return pf_out, prob_ds, (Q1, Q2), (ops1, ops2), (f1, f2)
