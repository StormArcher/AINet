#!/usr/bin/python3
#coding=utf-8
# change

import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
# from lib.pvt import PolypPVT
# from lib.pvt import PolypPVT2
from libTRM.pvt import TRM

from ptflops import get_model_complexity_info  # todo flops
import time   # todo fps


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, Softmax):
            pass
        elif isinstance(m, Sigmoid):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=1, dilation=1)  # todo 22-22

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        # input of resnet | output of pvt
        # -out2.shape - torch.Size([32, 64, 88, 88])
        # -out3.shape - torch.Size([32, 128, 44, 44])
        # -out4.shape - torch.Size([32, 320, 22, 22])
        # -out5.shape - torch.Size([32, 512, 11, 11])

        # input x2-x5
        # chanel 64 128 320 512
        # size 88 44 22 11

        # need x2-x5
        # channel 64 128 256 512
        # size 88 88 44 22

        # x2-x2
        # x3-UP(x3)
        # x4-CH256(UP(x4))
        # x5-UP(x5)
        # out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 160
        out1 = x
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)  # 88
        out2 = self.layer1(out1)  # in 64-88 out 88
        out3 = self.layer2(out2)  # in 88 out 44
        out4 = self.layer3(out3)  # in 44 out 22
        out5 = self.layer4(out4)  # in 22 out 11
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('/home/nk/zjc/pre-train/res/resnet50-19c8e357.pth'), strict=False)


class ResNet2(nn.Module):
    def __init__(self):
        super(ResNet2, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=1, dilation=1)  # todo 22-22

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        # out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('/home/nk/zjc/pre-train/res/resnet50-19c8e357.pth'), strict=False)


# =====>CBS<=====
class CBS(nn.Module):
    def __init__(self, inc=64, outc=64, kernel=3, padding=1, dilation=1):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel, stride=1, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(outc)

    def forward(self, x):
        x = F.sigmoid(self.bn(self.conv(x)))
        return x

    def initialize(self):
        weight_init(self)


# =====>CBR<=====
class CBR(nn.Module):
    def __init__(self, inc=64, outc=64, kernel=3, padding=1, dilation=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel, stride=1, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(outc)

    def forward(self, x):
        x =self.bn(self.conv(x))
        x = F.relu(x, inplace=True)
        return x

    def initialize(self):
        weight_init(self)


# =====>CM2<=====
class CombinationModule2(nn.Module):
    def __init__(self,):
        super(CombinationModule2, self).__init__()
        outc = 64
        self.CBS4 = CBS(inc=outc*4, outc=outc)
        self.CBS3 = CBS(inc=outc*3, outc=outc)
        self.CBS2 = CBS(inc=outc*2, outc=outc)

    def forward(self, out2r, out3r, out4r, out5r):
        S = out2r  # 2
        if out3r.size()[2:] != S.size()[2:]:
            out3r = F.interpolate(out3r, size=S.size()[2:], mode='bilinear')
            out4r = F.interpolate(out4r, size=S.size()[2:], mode='bilinear')
            out5r = F.interpolate(out5r, size=S.size()[2:], mode='bilinear')
        R2a = self.CBS4(torch.cat((out2r, out3r, out4r, out5r), 1))
        R2b = self.CBS3(torch.cat((out2r, out3r, out4r), 1))
        R2c = self.CBS2(torch.cat((out2r, out3r), 1))

        return R2a, R2b, R2c

    def initialize(self):
        weight_init(self)


# =====>CM3<=====
class CombinationModule3(nn.Module):
    def __init__(self,):
        super(CombinationModule3, self).__init__()
        outc = 64
        self.CBS4 = CBS(inc=outc*4, outc=outc)
        self.CBS3 = CBS(inc=outc*3, outc=outc)
        self.CBS2 = CBS(inc=outc*2, outc=outc)

    def forward(self, out2r, out3r, out4r, out5r):
        S = out3r  # 3
        if out2r.size()[2:] != S.size()[2:]:
            out2r = F.interpolate(out2r, size=S.size()[2:], mode='bilinear')
            out4r = F.interpolate(out4r, size=S.size()[2:], mode='bilinear')
            out5r = F.interpolate(out5r, size=S.size()[2:], mode='bilinear')
        R2a = self.CBS4(torch.cat((out2r, out3r, out4r, out5r), 1))
        R2b = self.CBS3(torch.cat((out2r, out3r, out4r), 1))
        R2c = self.CBS2(torch.cat((out2r, out3r), 1))

        return R2a, R2b, R2c

    def initialize(self):
        weight_init(self)


# =====>Flatten<=====
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    def initialize(self):
        weight_init(self)


# ===== ===== ===== ===== ===== =====>MIM<===== ===== ===== ===== ===== =====\
# =====>SIU<=====
class SIU(nn.Module):
    def __init__(self, ):
        super(SIU, self).__init__()
        outc = 64
        self.CBS1 = CBS(inc=outc, outc=outc)
        self.CBS2 = CBS(inc=outc, outc=outc)

        self.CBRadd1 = CBR(inc=outc, outc=outc)
        self.CBRadd2 = CBR(inc=outc, outc=outc)

        self.CBRmul1 = CBR(inc=outc, outc=outc)
        self.CBRmul2 = CBR(inc=outc, outc=outc)

        self.CBR = CBR(inc=outc * 2, outc=outc)

    def forward(self, R, D):
        Rs = self.CBS1(R)
        Ds = self.CBS2(D)

        CatR = self.CBRadd1(R + self.CBRmul1(R * Ds))
        CatD = self.CBRadd2(D + self.CBRmul2(D * Rs))

        F = torch.cat((CatR, CatD), 1)
        F = self.CBR(F)
        return F

    def initialize(self):
        weight_init(self)


# =====>CFE<=====
class CFE(nn.Module):
    def __init__(self,):
        super(CFE, self).__init__()
        outc = 64
        self.atrous3 = CBR(inc=outc, outc=outc, kernel=3, dilation=3, padding=3)
        self.atrous4 = CBR(inc=outc, outc=outc, kernel=3, dilation=5, padding=5)
        self.atrous5 = CBR(inc=outc, outc=outc, kernel=3, dilation=7, padding=7)
        self.CBRadd = CBR(inc=outc, outc=outc)
        self.CBRmul = CBR(inc=outc, outc=outc)
        self.CBS = CBS(inc=outc*3, outc=outc)

    def forward(self, x):
        A3 = self.atrous3(x)
        A5 = self.atrous4(x)
        A7 = self.atrous5(x)
        att = self.CBS(torch.cat((A3, A5, A7), 1))  # 22 18 12
        x = self.CBRadd(self.CBRmul(x*att) + att)
        return x

    def initialize(self):
        weight_init(self)


# =====>SRU<=====
class SRU(nn.Module):
    def __init__(self,):
        super(SRU, self).__init__()
        outc = 64
        self.CFE1 = CFE()
        self.CFE2 = CFE()
        self.CBR = CBR(inc=outc*2, outc=outc)

    def forward(self, R, D):
        R = self.CFE1(R)
        D = self.CFE2(D)
        F = torch.cat((R, D), 1)
        F = self.CBR(F)
        return F

    def initialize(self):
        weight_init(self)


# =====>MIM<=====
class MIM(nn.Module):
    def __init__(self,):
        super(MIM, self).__init__()
        self.SIU = SIU()
        self.SRU = SRU()

    def forward(self, R, D):
        Fa = self.SIU(R, D)  # spatial fusion part
        Fb = self.SRU(R, D)  # perception and reinforcement part
        F = Fa + Fb
        return F

    def initialize(self):
        weight_init(self)
# ===== ===== ===== ===== ===== =====>MIM<===== ===== ===== ===== ===== =====/


# ===== ===== ===== ===== ===== =====>MIMQK<===== ===== ===== ===== ===== =====\
# =====>CFE<=====
class CFE32(nn.Module):
    def __init__(self,):
        super(CFE32, self).__init__()
        outc = 32
        self.atrous3 = CBR(inc=outc, outc=outc, kernel=3, dilation=3, padding=3)
        self.atrous4 = CBR(inc=outc, outc=outc, kernel=3, dilation=5, padding=5)
        self.atrous5 = CBR(inc=outc, outc=outc, kernel=3, dilation=7, padding=7)
        self.CBRadd = CBR(inc=outc, outc=outc)
        self.CBRmul = CBR(inc=outc, outc=outc)
        self.CBS = CBS(inc=outc*3, outc=outc)

    def forward(self, x):
        A3 = self.atrous3(x)
        A5 = self.atrous4(x)
        A7 = self.atrous5(x)
        att = self.CBS(torch.cat((A3, A5, A7), 1))  # 22 18 12
        x = self.CBRadd(self.CBRmul(x*att) + att)
        return x

    def initialize(self):
        weight_init(self)


# =====>SRU32<=====
class SRU32(nn.Module):
    def __init__(self,):
        super(SRU32, self).__init__()
        outc = 64
        self.CBR32r = CBR(inc=outc, outc=32)
        self.CBR32d = CBR(inc=outc, outc=32)
        self.CFE1 = CFE32()
        self.CFE2 = CFE32()
        self.CBR32_r = CBR(inc=32, outc=32)
        self.CBR32_d = CBR(inc=32, outc=32)

    def forward(self, R, D):
        R = self.CBR32r(R)  # 64-32
        D = self.CBR32d(D)
        R = self.CBR32_r(self.CFE1(R))
        D = self.CBR32_r(self.CFE2(D))

        F = torch.cat((R, D), 1)
        return F

    def initialize(self):
        weight_init(self)


class QK_PAM(nn.Module):
    def __init__(self):
        super(QK_PAM, self).__init__()
        dim = 64
        # DQ
        self.DQ_conv1 = Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)  # in_dim//8
        # RK
        self.RK_conv1 = Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)  # in_dim//8
        # RV
        # self.RV_conv1 = Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        # self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, R, D):  # K, Q
        # ===> size
        B, C, H, W = R.size()
        # att1 RR
        DQ1 = self.DQ_conv1(D).view(B, -1, W * H).permute(0, 2, 1)  # (H*w, C)
        RK1 = self.RK_conv1(R).view(B, -1, W * H)  # (C, H*W)
        # RV1 = self.RV_conv1(R).view(m_batchsize, -1, width * height)  # (C, H*W)

        DR_energy = torch.bmm(DQ1, RK1)  # (H*W, H*W)
        DR_attention = self.softmax(DR_energy)

        # RRRout = torch.bmm(RV1, DR_attention.permute(0, 2, 1))  # (C, H*W), (H*W, H*W) -> (C, H*W)
        # RRRout = RRRout.view(m_batchsize, C, height, width)

        # RRRout = RRRout + R
        return DR_attention

    def initialize(self):
        weight_init(self)


class V_PAM(nn.Module):
    def __init__(self):
        super(V_PAM, self).__init__()
        dim = 64
        self.RV_conv1 = Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)

    def forward(self, R):  # K, Q
        # ===> size
        B, C, H, W = R.size()
        RV1 = self.RV_conv1(R).view(B, -1, W * H)  # (C, H*W)
        return RV1

    def initialize(self):
        weight_init(self)


# =====>MIMQK<=====
class MIMQK(nn.Module):
    def __init__(self,):
        super(MIMQK, self).__init__()
        cha = 64
        # generate FaFb
        self.SIU = SIU()
        self.SRU = SRU()
        self.QKS_s = QK_PAM()
        self.QKS_c = QK_PAM()
        self.VS_s = V_PAM()
        self.VS_c = V_PAM()

        # NL-att
        self.CBRatt1 = CBR()
        self.CBRatt2 = CBR()
        self.CBRresatt1 = CBR()
        self.CBRresatt2 = CBR()

        # NL-mlp
        self.CBRs1 = CBR(inc=cha, outc=cha // 2)
        self.CBRs2 = CBR(inc=cha // 2, outc=cha)
        self.CBRc1 = CBR(inc=cha, outc=cha // 2)
        self.CBRc2 = CBR(inc=cha // 2, outc=cha)

        self.CBRresmlp1 = CBR()
        self.CBRresmlp2 = CBR()

        # TRM-att-mlp
        self.TRM = TRM()
        self.CBRresmlp11 = CBR()
        self.CBRresmlp22 = CBR()

        # =>NL-combine
        self.CBRNL = CBR()
        # =>TRM-combine
        self.CBRTRM = CBR()
        # =>Final-combine
        self.CBRNLTRM = CBR(inc=64 * 2, outc=64)

    def forward(self, R, D):
        # =>Generate FaFb
        Fa = self.SIU(R, D)  # spatial fusion part
        Fb = self.SRU(R, D)

        # =>1 NL-att
        As = self.QKS_s(Fa, Fa)
        Vc = self.VS_c(Fa)

        Ac = self.QKS_c(Fb, Fb)
        Vs = self.VS_s(Fb)

        B, C, H, W = R.size()

        Ts = torch.bmm(Vs, As.permute(0, 2, 1))  # (C, H*W), (H*W, H*W) -> (C, H*W)
        Ts = Ts.view(B, C, H, W)

        Tc = torch.bmm(Vc, Ac.permute(0, 2, 1))  # (C, H*W), (H*W, H*W) -> (C, H*W)
        Tc = Tc.view(B, C, H, W)

        Tc = self.CBRatt1(Tc)
        Ts = self.CBRatt2(Ts)

        Ts = self.CBRresatt1(Ts+Fa)
        Tc = self.CBRresatt2(Tc+Fb)

        # =>2 NL-mlp
        Ts_ = Ts
        Tc_ = Tc

        Tc = self.CBRc2(self.CBRc1(Tc))  # res
        Ts = self.CBRs2(self.CBRs1(Ts))  # res

        Tc = self.CBRresmlp1(Tc+Tc_)
        Ts = self.CBRresmlp2(Ts+Ts_)

        # =>3 NL-combine
        F_NL = self.CBRNL(Ts + Tc)

        # =>1 TRM-att-mlp
        trma, trmb = self.TRM(Fa, Fb)
        trma = self.CBRresmlp11(trma)
        trmb = self.CBRresmlp22(trmb)

        # =>2 TRM-combine
        F_TRM = self.CBRTRM(trma + trmb)

        # =>Final-combine
        T = self.CBRNLTRM(torch.cat((F_NL, F_TRM), 1))
        return T

    def initialize(self):
        weight_init(self)
# ===== ===== ===== ===== ===== =====>MIMQK<===== ===== ===== ===== ===== =====/


# ===== ===== ===== ===== ===== =====>RCDB <===== ===== ===== ===== ===== =====\
# =====>CM4<=====
class CombinationModule4(nn.Module):
    def __init__(self,):
        super(CombinationModule4, self).__init__()
        outc = 64
        self.CBS4 = CBS(inc=outc*4, outc=outc)

    def forward(self, out2r, out3r, out4r, out5r):
        S = out4r  # 4
        if out2r.size()[2:] != S.size()[2:]:
            out2r = F.interpolate(out2r, size=S.size()[2:], mode='bilinear')
            out3r = F.interpolate(out3r, size=S.size()[2:], mode='bilinear')
            out5r = F.interpolate(out5r, size=S.size()[2:], mode='bilinear')
        R2a = self.CBS4(torch.cat((out2r, out3r, out4r, out5r), 1))
        return R2a

    def initialize(self):
        weight_init(self)


# =====>CM5<=====
class CombinationModule5(nn.Module):
    def __init__(self,):
        super(CombinationModule5, self).__init__()
        outc = 64
        self.CBS4 = CBS(inc=outc*4, outc=outc)

    def forward(self, out2r, out3r, out4r, out5r):
        S = out5r  # 5
        if out2r.size()[2:] != S.size()[2:]:
            out2r = F.interpolate(out2r, size=S.size()[2:], mode='bilinear')
            out3r = F.interpolate(out3r, size=S.size()[2:], mode='bilinear')
            out4r = F.interpolate(out4r, size=S.size()[2:], mode='bilinear')
        R2a = self.CBS4(torch.cat((out2r, out3r, out4r, out5r), 1))
        return R2a  # , R2b, R2c

    def initialize(self):
        weight_init(self)


# =====>RCDB<=====
class RCDB(nn.Module):
    def __init__(self,):
        super(RCDB, self).__init__()
        self.CM4 = CombinationModule4()
        self.CM5 = CombinationModule5()

    def forward(self, out2r, out3r, out4r, out5r):

        R2_ = out2r
        R3_ = out3r
        R4a = self.CM4(out2r, out3r, out4r, out5r)
        R5a = self.CM5(out2r, out3r, out4r, out5r)
        return R2_, R3_, R4a, R5a

    def initialize(self):
        weight_init(self)
# ===== ===== ===== ===== ===== =====>RCDB <===== ===== ===== ===== ===== =====/


# ===== ===== ===== ===== ===== =====>DecoderFPN <===== ===== ===== ===== ===== =====\
# =====>FPN<=====
class FPN(nn.Module):
    def __init__(self, ):
        super(FPN, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, L, H):
        if L.size()[2:] != H.size()[2:]:
            H = F.interpolate(H, size=L.size()[2:], mode='bilinear')
        L = L + H
        L = self.bn(self.conv(L))
        L = F.relu(L, inplace=True)
        return L

    def initialize(self):
        weight_init(self)


class DecoderFPN(nn.Module):
    def __init__(self,):
        super(DecoderFPN, self).__init__()
        self.fpn4 = FPN()
        self.fpn3 = FPN()
        self.fpn2 = FPN()

    def forward(self, F2, F3, F4, F5):
        fpn4 = self.fpn4(F4, F5)
        fpn3 = self.fpn3(F3, fpn4)
        fpn2 = self.fpn2(F2, fpn3)

        return fpn2

    def initialize(self):
        weight_init(self)
# ===== ===== ===== ===== ===== =====>DecoderFPN <===== ===== ===== ===== ===== =====/


class AINet(nn.Module):  # todo ->2
    def __init__(self, cfg):
        super(AINet, self).__init__()
        self.cfg      = cfg
        # <==1 backbone==>
        self.convbk = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bnbk = nn.BatchNorm2d(64)
        self.bkbone   = ResNet()

        self.convbk2 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bnbk2 = nn.BatchNorm2d(64)
        self.bkbone2   = ResNet2()

        # <==2 Transition layer==>
        # A squeeze
        outc = 64

        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, outc, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, outc, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(512, outc, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(256, outc, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))

        self.e_squeeze5 = nn.Sequential(nn.Conv2d(2048, outc, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.e_squeeze4 = nn.Sequential(nn.Conv2d(1024, outc, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.e_squeeze3 = nn.Sequential(nn.Conv2d(512, outc, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.e_squeeze2 = nn.Sequential(nn.Conv2d(256, outc, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))

        # <==3 MIMs==>
        self.MIM2 = MIM()
        self.MIM3 = MIM()
        self.MIM4 = MIM()
        self.MIMQK5 = MIMQK()

        # <==4 UNet branch==>
        self.CBRD4 = CBR(inc=outc*2, outc=outc)
        self.CBRD3 = CBR(inc=outc*2, outc=outc)
        self.CBRD2 = CBR(inc=outc*2, outc=outc)
        self.CBR = CBR(inc=outc, outc=outc)

        # <==5 RCDB==>
        self.RCDB = RCDB()

        # <==6 FPN==>
        self.DecoderFPN = DecoderFPN()

        self.linearp = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x, depth, shape=None):
        # <==1 backbone==>
        R = self.bnbk(self.convbk(x))
        R = F.relu(R, inplace=True)
        out2r, out3r, out4r, out5r = self.bkbone(R)  # ResNet()

        depth = self.bnbk2(self.convbk2((depth)))
        depth = F.relu(depth, inplace=True)
        out2d, out3d, out4d, out5d = self.bkbone2(depth)  # ResNet()

        # <==2 Trasmit layer==>
        out2r, out3r, out4r, out5r = \
            self.squeeze2(out2r), self.squeeze3(out3r), self.squeeze4(out4r), self.squeeze5(out5r)

        out2d, out3d, out4d, out5d = \
            self.e_squeeze2(out2d), self.e_squeeze3(out3d), self.e_squeeze4(out4d), self.e_squeeze5(out5d)

        # <==3 MIMs==>
        F5 = self.MIMQK5(out5r, out5d)  # att5
        F4 = self.MIM4(out4r, out4d)  # att4, v5
        F3 = self.MIM3(out3r, out3d)  # v4/down2
        F2 = self.MIM2(out2r, out2d)

        # <==4 UNet branch==>
        if F5.size()[2:] != F4.size()[2:]:
            F5 = F.interpolate(F5, size=F4.size()[2:], mode='bilinear')
        D4 = self.CBRD4(torch.cat((F4, F5), 1))  # cat F5-F4-T5
        D4_ = D4
        D4 = F.interpolate(D4, size=F3.size()[2:], mode='bilinear')
        D3 = self.CBRD3(torch.cat((F3, D4), 1))  # cat F3-D4-T4
        D3_ = D3
        D3 = F.interpolate(D3, size=F2.size()[2:], mode='bilinear')
        D2 = self.CBRD2(torch.cat((F2, D3), 1))

        # <==5 RCDB==>
        F2_, F3_, F4_, F5_ = self.RCDB(D2, D3_, D4_, F5)  # todo B D2345

        # # <==6 FPN branch==>
        outF = self.DecoderFPN(F2_, F3_, F4_, F5_)
        output = self.CBR(outF+D2)

        shape = x.size()[2:] if shape is None else shape

        out = F.interpolate(self.linearp(output), size=shape, mode='bilinear')

        return out

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
