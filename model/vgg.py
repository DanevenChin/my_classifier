# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2021/10/8 19:03
@file    : vgg.py
@desc    : 
"""
import torch
import torch.nn as nn


def conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return layer


def fc_layer(in_features, out_features):
    layer = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU()
    )
    return layer


class VGG16(nn.Module):
    def __init__(self, num_classes=1000, img_size=224):
        super(VGG16, self).__init__()
        self.conv_1 = conv_layer(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = conv_layer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3 = conv_layer(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_4 = conv_layer(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_5 = conv_layer(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_6 = conv_layer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_7 = conv_layer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_8 = conv_layer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_9 = conv_layer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_10 = conv_layer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_11 = conv_layer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_12 = conv_layer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_13 = conv_layer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.maxpool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        final_size = img_size // (2**5)
        self.fc_1 = fc_layer(in_features=final_size*final_size*512, out_features=4096)
        self.fc_2 = fc_layer(in_features=4096, out_features=4096)
        self.fc_3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)

        x = self.maxpool_1(x)
        x = self.conv_3(x)
        x = self.conv_4(x)

        x = self.maxpool_2(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)

        x = self.maxpool_3(x)
        x = self.conv_8(x)
        x = self.conv_9(x)
        x = self.conv_10(x)

        x = self.maxpool_4(x)
        x = self.conv_11(x)
        x = self.conv_12(x)
        x = self.conv_13(x)

        x = self.maxpool_5(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x

if __name__ == '__main__':
    x = torch.randn((2,3,512,512))
    vgg_16 = VGG16(num_classes=100, img_size=x.size(-1))
    y = vgg_16(x)
    print(y.shape)