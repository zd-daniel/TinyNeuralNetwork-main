import torch
import torch.nn as nn
import torch.nn.functional as F

import os

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
DEFAULT_STATE_DICT = os.path.join(CURRENT_PATH, "mobilenet.pth")
DEFAULT_MODEL = os.path.join(CURRENT_PATH, "mobilenet.pt")


class Mobilenet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_0_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                   dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_0_1 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model_0_2 = nn.ReLU(True)
        self.model_1_0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                   dilation=(1, 1), groups=32, bias=False, padding_mode='zeros')
        self.model_1_1 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model_1_2 = nn.ReLU(True)
        self.model_1_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_1_4 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model_1_5 = nn.ReLU(True)
        self.model_2_0 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                   dilation=(1, 1), groups=64, bias=False, padding_mode='zeros')
        self.model_2_1 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model_2_2 = nn.ReLU(True)
        self.model_2_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_2_4 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_2_5 = nn.ReLU(True)
        self.model_3_0 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                   dilation=(1, 1), groups=128, bias=False, padding_mode='zeros')
        self.model_3_1 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_3_2 = nn.ReLU(True)
        self.model_3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_3_4 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_3_5 = nn.ReLU(True)
        self.model_4_0 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                   dilation=(1, 1), groups=128, bias=False, padding_mode='zeros')
        self.model_4_1 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_4_2 = nn.ReLU(True)
        self.model_4_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_4_4 = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_4_5 = nn.ReLU(True)
        self.model_5_0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                   dilation=(1, 1), groups=256, bias=False, padding_mode='zeros')
        self.model_5_1 = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_5_2 = nn.ReLU(True)
        self.model_5_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_5_4 = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_5_5 = nn.ReLU(True)
        self.model_6_0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                   dilation=(1, 1), groups=256, bias=False, padding_mode='zeros')
        self.model_6_1 = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_6_2 = nn.ReLU(True)
        self.model_6_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_6_4 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_6_5 = nn.ReLU(True)
        self.model_7_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                   dilation=(1, 1), groups=512, bias=False, padding_mode='zeros')
        self.model_7_1 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_7_2 = nn.ReLU(True)
        self.model_7_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_7_4 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_7_5 = nn.ReLU(True)
        self.model_8_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                   dilation=(1, 1), groups=512, bias=False, padding_mode='zeros')
        self.model_8_1 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_8_2 = nn.ReLU(True)
        self.model_8_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_8_4 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_8_5 = nn.ReLU(True)
        self.model_9_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                   dilation=(1, 1), groups=512, bias=False, padding_mode='zeros')
        self.model_9_1 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_9_2 = nn.ReLU(True)
        self.model_9_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_9_4 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.model_9_5 = nn.ReLU(True)
        self.model_10_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
                                    padding=(1, 1), dilation=(1, 1), groups=512, bias=False, padding_mode='zeros')
        self.model_10_1 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                         track_running_stats=True)
        self.model_10_2 = nn.ReLU(True)
        self.model_10_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_10_4 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                         track_running_stats=True)
        self.model_10_5 = nn.ReLU(True)
        self.model_11_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
                                    padding=(1, 1), dilation=(1, 1), groups=512, bias=False, padding_mode='zeros')
        self.model_11_1 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                         track_running_stats=True)
        self.model_11_2 = nn.ReLU(True)
        self.model_11_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_11_4 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                         track_running_stats=True)
        self.model_11_5 = nn.ReLU(True)
        self.model_12_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2),
                                    padding=(1, 1), dilation=(1, 1), groups=512, bias=False, padding_mode='zeros')
        self.model_12_1 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                                         track_running_stats=True)
        self.model_12_2 = nn.ReLU(True)
        self.model_12_3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_12_4 = nn.BatchNorm2d(num_features=1024, eps=1e-05, momentum=0.1, affine=True,
                                         track_running_stats=True)
        self.model_12_5 = nn.ReLU(True)
        self.model_13_0 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1),
                                    padding=(1, 1), dilation=(1, 1), groups=1024, bias=False, padding_mode='zeros')
        self.model_13_1 = nn.BatchNorm2d(num_features=1024, eps=1e-05, momentum=0.1, affine=True,
                                         track_running_stats=True)
        self.model_13_2 = nn.ReLU(True)
        self.model_13_3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.model_13_4 = nn.BatchNorm2d(num_features=1024, eps=1e-05, momentum=0.1, affine=True,
                                         track_running_stats=True)
        self.model_13_5 = nn.ReLU(True)
        self.model_14 = nn.AvgPool2d(kernel_size=7, stride=7, padding=0, ceil_mode=False, count_include_pad=True,
                                     divisor_override=None)
        self.fc = nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, input_1):
        model_0_0 = self.model_0_0(input_1)
        model_0_1 = self.model_0_1(model_0_0)
        model_0_2 = self.model_0_2(model_0_1)
        model_1_0 = self.model_1_0(model_0_2)
        model_1_1 = self.model_1_1(model_1_0)
        model_1_2 = self.model_1_2(model_1_1)
        model_1_3 = self.model_1_3(model_1_2)
        model_1_4 = self.model_1_4(model_1_3)
        model_1_5 = self.model_1_5(model_1_4)
        model_2_0 = self.model_2_0(model_1_5)
        model_2_1 = self.model_2_1(model_2_0)
        model_2_2 = self.model_2_2(model_2_1)
        model_2_3 = self.model_2_3(model_2_2)
        model_2_4 = self.model_2_4(model_2_3)
        model_2_5 = self.model_2_5(model_2_4)
        model_3_0 = self.model_3_0(model_2_5)
        model_3_1 = self.model_3_1(model_3_0)
        model_3_2 = self.model_3_2(model_3_1)
        model_3_3 = self.model_3_3(model_3_2)
        model_3_4 = self.model_3_4(model_3_3)
        model_3_5 = self.model_3_5(model_3_4)
        model_4_0 = self.model_4_0(model_3_5)
        model_4_1 = self.model_4_1(model_4_0)
        model_4_2 = self.model_4_2(model_4_1)
        model_4_3 = self.model_4_3(model_4_2)
        model_4_4 = self.model_4_4(model_4_3)
        model_4_5 = self.model_4_5(model_4_4)
        model_5_0 = self.model_5_0(model_4_5)
        model_5_1 = self.model_5_1(model_5_0)
        model_5_2 = self.model_5_2(model_5_1)
        model_5_3 = self.model_5_3(model_5_2)
        model_5_4 = self.model_5_4(model_5_3)
        model_5_5 = self.model_5_5(model_5_4)
        model_6_0 = self.model_6_0(model_5_5)
        model_6_1 = self.model_6_1(model_6_0)
        model_6_2 = self.model_6_2(model_6_1)
        model_6_3 = self.model_6_3(model_6_2)
        model_6_4 = self.model_6_4(model_6_3)
        model_6_5 = self.model_6_5(model_6_4)
        model_7_0 = self.model_7_0(model_6_5)
        model_7_1 = self.model_7_1(model_7_0)
        model_7_2 = self.model_7_2(model_7_1)
        model_7_3 = self.model_7_3(model_7_2)
        model_7_4 = self.model_7_4(model_7_3)
        model_7_5 = self.model_7_5(model_7_4)
        model_8_0 = self.model_8_0(model_7_5)
        model_8_1 = self.model_8_1(model_8_0)
        model_8_2 = self.model_8_2(model_8_1)
        model_8_3 = self.model_8_3(model_8_2)
        model_8_4 = self.model_8_4(model_8_3)
        model_8_5 = self.model_8_5(model_8_4)
        model_9_0 = self.model_9_0(model_8_5)
        model_9_1 = self.model_9_1(model_9_0)
        model_9_2 = self.model_9_2(model_9_1)
        model_9_3 = self.model_9_3(model_9_2)
        model_9_4 = self.model_9_4(model_9_3)
        model_9_5 = self.model_9_5(model_9_4)
        model_10_0 = self.model_10_0(model_9_5)
        model_10_1 = self.model_10_1(model_10_0)
        model_10_2 = self.model_10_2(model_10_1)
        model_10_3 = self.model_10_3(model_10_2)
        model_10_4 = self.model_10_4(model_10_3)
        model_10_5 = self.model_10_5(model_10_4)
        model_11_0 = self.model_11_0(model_10_5)
        model_11_1 = self.model_11_1(model_11_0)
        model_11_2 = self.model_11_2(model_11_1)
        model_11_3 = self.model_11_3(model_11_2)
        model_11_4 = self.model_11_4(model_11_3)
        model_11_5 = self.model_11_5(model_11_4)
        model_12_0 = self.model_12_0(model_11_5)
        model_12_1 = self.model_12_1(model_12_0)
        model_12_2 = self.model_12_2(model_12_1)
        model_12_3 = self.model_12_3(model_12_2)
        model_12_4 = self.model_12_4(model_12_3)
        model_12_5 = self.model_12_5(model_12_4)
        model_13_0 = self.model_13_0(model_12_5)
        model_13_1 = self.model_13_1(model_13_0)
        model_13_2 = self.model_13_2(model_13_1)
        model_13_3 = self.model_13_3(model_13_2)
        model_13_4 = self.model_13_4(model_13_3)
        model_13_5 = self.model_13_5(model_13_4)
        model_14 = self.model_14(model_13_5)
        view_1 = model_14.view(model_14.shape[0], -1)
        fc = self.fc(view_1)
        return fc


if __name__ == "__main__":
    model = Mobilenet()
    model.load_state_dict(torch.load(DEFAULT_STATE_DICT))
    torch.save(model, DEFAULT_MODEL)

    model.eval()
    model.cpu()

    dummy_input_0 = torch.ones((1, 3, 224, 224))

    output = model(dummy_input_0)
    print(output)
