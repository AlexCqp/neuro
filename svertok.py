import math
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
class ResNet18(nn.Module):
    def __init__(self, out_dim):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.loc_conv = nn.Conv2d(512, 64, kernel_size=3,stride=1, padding=1)
        self.loc_fc = nn.Sequential(
        nn.Linear(7 * 7 * 64, 512),
        nn.LeakyReLU(),
        nn.Linear(512, out_dim, bias=True))
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes,kernel_size=1, stride=stride,bias=False),nn.BatchNorm2d(planes),)
        layers = [resnet.BasicBlock(self.inplanes, planes,stride, downsample)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(resnet.BasicBlock(self.inplanes,planes))
        return nn.Sequential(*layers)
    def forward(self, x, return_feature_vector=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # localization
        loc_res = self.loc_conv(x)
        loc_res = nn.functional.relu(loc_res)
        loc_feature_vector = loc_res
        loc_res = loc_res.flatten(1)
        loc_res = self.loc_fc(loc_res)
        if return_feature_vector:
            return loc_res, loc_feature_vector
        else:
            return loc_res
