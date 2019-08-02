import numpy as np
from scipy import interpolate

import torch
from torch import nn
from torch.nn import functional as F


class Upsampler(nn.Module):

    def __init__(self):
        super(Upsampler, self).__init__()

        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 1, 5, padding=(0, 2))
        self.hidden_act = F.relu
        self.output_act = torch.sigmoid

    def forward(self, input):
        x = self.upsample(input)
        x = self.conv1(x)
        x = self.hidden_act(x)

        x = self.upsample(x)
        x = self.conv2(x)
        x = self.hidden_act(x)

        x = self.upsample(x)
        x = self.conv3(x)
        x = self.output_act(x * 2)

        return x
