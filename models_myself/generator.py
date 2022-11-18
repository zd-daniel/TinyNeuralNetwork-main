# -*- coding: utf-8 -*-
# @Time    : 2022/5/8 9:48
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : generator.py
# @Software: PyCharm
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_myself.darknet import Darknet, CSPDarknet
from models_myself.network_blocks import BaseConv, DWConv, SelfAttention, CustomPixelShuffle_ICNR, CSPLayer
from models_myself.model_utils import initialize_weights


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=0.5,
        in_features=("stem", "dark2", "dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,  # 深度可分离卷积
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act, out_features=in_features)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x4, x3, x2, x1, x0] = features
        x5 = nn.MaxPool2d(2, 2)(input)

        # 以下8行是FPN
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        # 以下6行是PAN
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (x5, x4, x3, pan_out2, pan_out1, pan_out0)
        return outputs


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()

        self.global_level = nn.Sequential(
            DWConv(512, 512, 3, stride=1, pad=1, act="lrelu"),
            DWConv(512, 512, 4, stride=2, act="lrelu"),
            DWConv(512, 512, 3, stride=1, pad=1, act="lrelu"),
            DWConv(512, 512, 4, stride=2, pad=0, act="lrelu"),
        )
        self.global_level_feature = nn.Sequential(
            BaseConv(512, 256, 1, stride=1, act="lrelu"),
            BaseConv(256, 128, 1, stride=1, act="lrelu"),
        )
        self.classification_level = nn.Sequential(
            BaseConv(512, 512, 1, stride=1, act="lrelu"),
            BaseConv(512, 512, 1, stride=1, act="lrelu"),
            nn.Conv2d(512, 365, (1, 1), (1, 1), (0, 0))
        )

    def forward(self, x):
        feature = self.global_level(x)

        feature = self.global_level_feature(feature)
        return feature


class Colorization(nn.Module):
    def __init__(self):
        super(Colorization, self).__init__()

        self.merge = nn.Sequential(
            DWConv(256, 256, 3, stride=1, act="lrelu"),
            # SelfAttention(256)
        )

        self.color_level1 = nn.Sequential(
            # CustomPixelShuffle_ICNR(256, 128, scale=2, blur=True),  # 算子不支持部署
            nn.Upsample(scale_factor=2),
            DWConv(256, 128, 3, stride=1, act="lrelu"),
        )
        self.color_level1_conv = nn.Sequential(
            DWConv(192, 128, 3, stride=1, act="lrelu"),
            DWConv(128, 128, 3, stride=1, act="lrelu"),
        )

        self.color_level2 = nn.Sequential(
            # CustomPixelShuffle_ICNR(128, 64, scale=2, blur=True),
            nn.Upsample(scale_factor=2),
            DWConv(128, 64, 3, stride=1, act="lrelu"),
        )
        self.color_level2_conv = nn.Sequential(
            DWConv(96, 64, 3, stride=1, act="lrelu"),
            DWConv(64, 64, 3, stride=1, act="lrelu"),
        )

        self.color_level3 = nn.Sequential(
            # CustomPixelShuffle_ICNR(64, 32, scale=2, blur=True),
            DWConv(64, 32, 3, stride=1, act="lrelu"),
        )
        self.color_level3_conv = nn.Sequential(
            DWConv(35, 32, 3, stride=1, act="lrelu"),
            DWConv(32, 32, 3, stride=1, act="lrelu"),
            nn.Conv2d(32, 3, (3, 3), (1, 1), (1, 1)),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, glo_repeat, mid, x3, x4, x5):
        out = self.merge(torch.cat((mid, glo_repeat), dim=1))

        out = self.color_level1(out)
        out = self.color_level1_conv(torch.cat((out, x3), dim=1))

        out = self.color_level2(out)
        out = self.color_level2_conv(torch.cat((out, x4), dim=1))

        out = self.color_level3(out)
        out = self.color_level3_conv(torch.cat((out, x5), dim=1))
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.low_level = YOLOPAFPN(depthwise=True)
        self.classification = Classification()
        self.colorization = Colorization()

        self.low_level.apply(initialize_weights)
        self.classification.apply(initialize_weights)
        self.colorization.apply(initialize_weights)

    def forward(self, x):
        # low阶段 + mid阶段
        feature = self.low_level(x)
        x5, x4, x3, mid, _, low = feature

        # global阶段 + 场景分类模块
        pooled_features = F.interpolate(low, size=(8, 8), mode='nearest')
        glo = self.classification(pooled_features)
        # glo_repeat = glo.repeat(1, 1, mid.shape[2], mid.shape[3])  # repeat操作在opencv读onnx时报错(不支持Tile)
        glo_repeat = F.interpolate(glo, (mid.shape[2], mid.shape[3]), mode='nearest')

        # colorization阶段
        color_yuv = self.colorization(glo_repeat, mid, x3, x4, x5)

        return color_yuv[:, 1:]


if __name__ == '__main__':
    image = torch.rand((2, 1, 512, 512)).cuda()
    model = Generator().cuda()
    torch.save(model.state_dict(), '../output/netG_latest.pth')
    from model_utils import fuse_model
    model = fuse_model(model)
    print(model)

    color, cls, low, mid, glo = model(image)
    print(color.shape)
    print(cls.shape)
    print(low.shape)
    print(mid.shape)
    print(glo.shape)

    num = 0
    for param in model.parameters():
        num += torch.prod(torch.tensor(list(param.shape)))
    print(num)
