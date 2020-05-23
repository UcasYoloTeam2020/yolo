import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F

__all__ = ['ResNet50', 'ResNet101', 'ResNet152']


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=1)

        self.dil_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=2, dilation=2)
        self.dil_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=4, dilation=4)
        self.dil_conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=8, dilation=8)
        self.dil_bn1 = nn.BatchNorm2d(512)
        self.dil_bn2 = nn.BatchNorm2d(512)
        self.dil_bn3 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512 * 3, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x1 = F.relu(self.dil_bn1(self.dil_conv1(x)))
        x2 = F.relu(self.dil_bn2(self.dil_conv2(x)))
        x3 = F.relu(self.dil_bn3(self.dil_conv3(x)))
        x = torch.cat((x1, x2, x3), 1)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        return x


def ResNet_dilat():
    return ResNet([3, 4, 6, 3])

# model=ResNet_dilat()
# from torchsummary import summary

# import pdb;pdb.set_trace()
