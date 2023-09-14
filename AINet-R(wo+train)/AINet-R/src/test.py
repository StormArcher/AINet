# !/usr/bin/python3
# coding=utf-8

import os
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net import AINet
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from ptflops import get_model_complexity_info  # todo flops
import time   # todo fps


num = '80'
class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset

        self.cfg = Dataset.Config(datapath=path, snapshot='./out/model-'+ num, mode='test')

        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        '''
        with torch.cuda.device(0):  # todo 1
            macs, params = get_model_complexity_info(self.net, (4, 352, 352), as_strings=True, print_per_layer_stat=True,
                                                     verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))'''

    def show(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                # 2, 4, 1, 4
                out2h_up2, out3h, out4h, out5v, e_out2h_up2, e_out3h, e_out4h, e_out5v, out2h_up1, e_out2h_up1 = self.net(image)
                # out = out2u

                plt.subplot(221)
                plt.imshow(np.uint8(image[0].permute(1, 2, 0).cpu().numpy() * self.cfg.std + self.cfg.mean))
                plt.subplot(222)
                plt.imshow(mask[0].cpu().numpy())
                plt.subplot(223)
                plt.imshow(out[0, 0].cpu().numpy())
                plt.subplot(224)
                plt.imshow(torch.sigmoid(out[0, 0]).cpu().numpy())
                plt.show()
                input()

    def save(self):
        with torch.no_grad():
            print('<=to show=> ', self.cfg.datapath.split('/')[-1])  # ECSSD
            print('=>loader', self.loader)
            ticks_back = 0

            idx = 0  # todo 1
            time_spent = []

            for image, mask, nir, shape, name in self.loader:


                image = image.cuda().float()
                nir = nir.cuda().float().unsqueeze(1)

                # <4,4,1>
                start_time = time.time()
                outs = self.net(image, nir, shape)
                '''
                shape = 352
                K = F.interpolate(e_out2h_up2, shape, mode='bilinear', align_corners=True)  # 放大到相应尺寸
                output = torch.squeeze(K, dim=0)  # (64, 128, 128)
                print(type(output), output.shape)
                output_arr = output.data.cpu().numpy()  # 换为cpu
                output_arr1 = np.mean(output_arr, axis=0)  #
                plt.axis('off')
                plt.imshow(output_arr1, cmap=plt.cm.jet)
                # you can change your path
                # head = '../eval/maps/try6_10_32_visual-CAMO-d'  '/' + self.cfg.datapath.split('/')[-1]
                # if not os.path.exists(head):
                # os.makedirs(head)
                # name1 = head+'/'+name[0]+'.png'
                name1 = './color/e_out2h_up2' + name[0] + '46-pred1.png'
                # name1 = './e_out2h_up2.png' + self.cfg.datapath.split('/')[-1]
                plt.savefig(name1, pad_inches=0)'''

                out2h_up2 = (torch.sigmoid(outs[0, 0]) * 255).cpu().numpy()
                # todo 4
                time_spent.append(time.time() - start_time)
                if idx % 100 == 0:
                    time_spent = []
                if idx % 99 == 0:
                    # print('time/FPS', np.mean(time_spent), 1 * 1 // np.mean(time_spent))
                    print(1 * 1 // np.mean(time_spent))
                idx = idx + 1
                # out3h = (torch.sigmoid(out3h[0, 0]) * 255).cpu().numpy()
                # out4h = (torch.sigmoid(out4h[0, 0]) * 255).cpu().numpy()
                # out5v = (torch.sigmoid(out5v[0, 0]) * 255).cpu().numpy()

                # e_out2h_up2 = (torch.sigmoid(e_out2h_up2[0, 0]) * 255).cpu().numpy()
                # e_out3h = (torch.sigmoid(e_out3h[0, 0]) * 255).cpu().numpy()
                # e_out4h = (torch.sigmoid(e_out4h[0, 0]) * 255).cpu().numpy()
                 #e_out5v = (torch.sigmoid(e_out5v[0, 0]) * 255).cpu().numpy()

                # out2h_up1 = (torch.sigmoid(out2h_up1[0, 0]) * 255).cpu().numpy()
                # e_out2h_up1 = (torch.sigmoid(e_out2h_up1[0, 0]) * 255).cpu().numpy()

                # out2h_mid = (torch.sigmoid(out2h_mid[0, 0]) * 255).cpu().numpy()
                head = '../eval/maps/AINet/' + self.cfg.datapath.split('/')[-1] + '-AINet-R-Confirm'+ num
                # head = '../eval/maps/F3Net/' + 'ECSSD'
                # print('--head--', head)
                if not os.path.exists(head):
                    os.makedirs(head)

                # print('xxx3', head + '/' + name[0] + '.png', np.round(pred))

                cv2.imwrite(head + '/' + name[0] + '.png', np.round(out2h_up2))
                # acv2.imwrite(head + '/' + name[0] + 'e.png', np.round(e_out2h_up2))


                # ticks = time.time()
                # fps = 1/(ticks - ticks_back)
                # ticks_back = ticks
                # print('time=>', ticks)
                # print('FPS=>', fps)
                # cv2.imwrite(head + '/' + name[0] + '_s3.png', np.round(out3h))
                # cv2.imwrite(head + '/' + name[0] + '_s4.png', np.round(out4h))
                # cv2.imwrite(head + '/' + name[0] + '_s5.png', np.round(out5v))

                # cv2.imwrite(head + '/' + name[0] + '_e2.png', np.round(e_out2h_up2))
                # cv2.imwrite(head + '/' + name[0] + '_e3.png', np.round(e_out3h))
                # cv2.imwrite(head + '/' + name[0] + '_e4.png', np.round(e_out4h))
                # cv2.imwrite(head + '/' + name[0] + '_e5.png', np.round(e_out5v))

                # cv2.imwrite(head + '/' + name[0] + '_mid.png', np.round(out2h_mid))


if __name__ == '__main__':
    # '../data/SOD', '../data/THUR15K'
    # '../data/ECSSD', '../data/PASCALS', '../data/DUTS-TE', '../data/HKU-IS', '../data/DUT-OMRON'

    '''
    '/data/home/scv1844/run/data/data-RGBD/test/LFSD',
                 '/data/home/scv1844/run/data/data-RGBD/test/NJU2K-Test',
                 '/data/home/scv1844/run/data/data-RGBD/test/SIP',
                 '/data/home/scv1844/run/data/data-RGBD/test/STEREO'
                 '''

    for path in [
                 # '/home/nk/zjc/data/data-RGBT/VT5000-Train',
                 '/home/aaa/DL/data/data-RGBT/VT821',
                 '/home/aaa/DL/data/data-RGBT/VT1000',
                 '/home/aaa/DL/data/data-RGBT/VT5000-Test'
                 ]:
        t = Test(dataset, AINet, path)
        t.save()
        # t.show()
