import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import pathlib
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
import time
import copy

class DoubleConv(nn.Module):

    def __init__(self, ch_in, ch_out, kernel_size=3):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(ch_out)
        )


    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, ch_in, ch_out, n_downsample=4):
        super(UNet, self).__init__()

        ch_in_down = [ch_in]+[64*(2**i) for i in range(n_downsample)]
        ch_out_down = [64]+[64*(2**(i+1)) for i in range(n_downsample)]
        ch_in_up = [1024//(2**i) for i in range(n_downsample)]
        ch_out_up = [512//(2**i) for i in range(n_downsample)]

        self.Down_conv = nn.ModuleList(
            [self.Get_DoubleConv(ch_in_down[i], ch_out_down[i]) for i in range(n_downsample+1)]
        )

        self.Up_conv = nn.ModuleList(
            [self.Get_DoubleConv(ch_in_up[i], ch_out_up[i]) for i in range(n_downsample)]
        )

        self.Up_samplers = nn.ModuleList(
            [self.Get_UpSampler(ch_in_up[i], ch_out_up[i]) for i in range(n_downsample)]
        )

        self.Last_conv = nn.Conv2d(ch_out_up[-1], ch_out, kernel_size=1)

    def Get_DoubleConv(self, ch_in, ch_out):
        return nn.Sequential(
            DoubleConv(ch_in, ch_out)
        )

    def Get_UpSampler(self, ch_in, ch_out):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        )

    def forward(self, x):
        feature_map = []
        for down_path in self.Down_conv:
            x = down_path(x)
            if len(feature_map) != len(self.Down_conv)-1:
                feature_map.append(x)
                x = F.max_pool2d(x, 2)

        for up_path, up_sample, f_map in zip(self.Up_conv, self.Up_samplers, feature_map[::-1]):
            x = up_sample(x)
            x = torch.cat([x, f_map], dim=1)
            x = up_path(x)

        x = self.Last_conv(x)

        return x