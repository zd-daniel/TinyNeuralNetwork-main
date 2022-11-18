# -*- coding: utf-8 -*-
# @Time    : 2022/5/8 22:55
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : discriminator.py
# @Software: PyCharm


import torch.nn as nn
from models_myself.model_utils import initialize_weights


class Discriminator(nn.Module):
    def __init__(self, input_shape=3, channel=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, (4, 4), stride=(2, 2), padding=(1, 1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_shape, channel, normalize=False),
            *discriminator_block(channel, channel * 2),
            *discriminator_block(channel * 2, channel * 4),
            *discriminator_block(channel * 4, channel * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(channel * 8, 1, (4, 4), padding=(1, 1))
        )

        self.model.apply(initialize_weights)

    def forward(self, img):
        return self.model(img)
