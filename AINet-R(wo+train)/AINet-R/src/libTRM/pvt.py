import torch
import torch.nn as nn
import torch.nn.functional as F
from libTRM.pvtv2 import TRMM
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt

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
        else:
            m.initialize()


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


#  可视化模块-即插即用
def visualize_feature_map(rank, attibute, feature_map, name):  # [out1, out2, out3, out4, out5], img_name, i
    print('=====>')

    # =>1 只留下C,H,W
    print('1-输入BCHW', feature_map.shape)
    feature_map = feature_map.squeeze(0)  # [1,64,88,88] -> [64,88,88]
    print('1-简化为CHW', feature_map.shape)

    # =>2 计算通道数C
    feature_map_combination = []
    num_pic = feature_map.shape[0]  # [64,88,88] -> 64
    print('2-通道数C', num_pic)
    # =>3 重新设计一个列表，里面放64张特征图[f1,..,f64]
    for i in range(0, num_pic):  # 循环各个通道
        # print('==feature_map.shape', feature_map.shape)
        feature_map_split = feature_map[i, :, :]
        feature_map_combination.append(feature_map_split)  # 添加列表

    # =>4 HW维度求和
    feature_map_sum = sum(ele for ele in feature_map_combination)  # 累加各个通道
    feature_map_sum = feature_map_sum.cuda().data.cpu()

    # =>5 保存
    plt.imshow(feature_map_sum)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #                             数据集sota1 + 图片名dog + 特征位置
    path ='/home/aaa/DL/code/CST/see/' + rank + attibute +'-'+ name+".png"
    print('4-图片路径', path)
    plt.savefig(path, bbox_inches='tight', dpi=320,
                pad_inches=0.0)


class TRM(nn.Module):
    def __init__(self, channel=32):
        super(TRM, self).__init__()

        self.trm = TRMM()

    def forward(self, a, b):

        f = self.trm(a, b)
        return f

    def initialize(self):
        self.load_state_dict(torch.load('/home/nk/zjc/pre-train/pvt/pvt_v2_b2.pth'), strict=False)


if __name__ == '__main__':
    model = PVTv2().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
