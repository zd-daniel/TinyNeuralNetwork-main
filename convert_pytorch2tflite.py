# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 17:06
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : convert_pytorch2tflite.py
# @Software: PyCharm


import os
import numpy as np
import torch

from models_myself.generator import Generator
from models_myself.model_utils import fuse_model

from tinynn.converter import TFLiteConverter


def demo():
    # Networks
    model = Generator().cpu()
    checkpoint = torch.load('models_myself/colorization.pth')
    model_dict = model.state_dict()
    checkpoint_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    model = fuse_model(model)
    model.eval()

    dummy_input = torch.rand((1, 3, 384, 512))

    output_path = os.path.join('models_myself', 'out', 'colorization.tflite')

    # When converting quantized models, please ensure the quantization backend is set.
    # torch.backends.quantized.engine = 'qnnpack'

    # The code section below is used to convert the model to the TFLite format
    # If you want perform dynamic quantization on the float models,
    # you may refer to `dynamic.py`, which is in the same folder.
    # As for static quantization (e.g. quantization-aware training and post-training quantization),
    # please refer to the code examples in the `examples/quantization` folder.
    converter = TFLiteConverter(model, dummy_input, output_path)
    converter.convert()

    # import cv2
    # from time import time
    # cap = cv2.VideoCapture('E:/ViewVideo/weiguang/data_collection/3/weiguang_1.mp4')
    # while cap.isOpened():
    #     ret, image = cap.read()
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     # cv2.imshow('image', image)
    #
    #     image_resize = cv2.resize(image, (640, 384)) / 255
    #     image_resize = np.repeat(image_resize[..., None], 3, axis=2)
    #
    #     image_input = torch.from_numpy(image_resize.transpose([2, 0, 1])[None].astype('float32')).cuda()
    #     start_time = time()
    #     fake, _, _, _, _ = model(image_input)
    #     end_time = time()
    #     print('time: ', (end_time - start_time) * 1000)
    #     fake = fake.clip(0, 1)
    #     fake = (fake[0].permute(1, 2, 0).detach().cpu().numpy() * 255 + 0.5).astype('uint8')
    #     fake = cv2.resize(fake, (image.shape[1], image.shape[0]))
    #     gen_bgr = cv2.cvtColor(np.concatenate((image[..., None], fake[..., 1:]), axis=2), cv2.COLOR_YUV2BGR)
    #     cv2.imshow('gen', gen_bgr)
    #     cv2.waitKey(1)


if __name__ == '__main__':
    demo()
