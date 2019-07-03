import numpy as np
from scipy import interpolate

import torch
from torch import nn
from torch.nn import functional as F


class Upsampler(nn.Module):

    def __init__(self):
        super(Upsampler, self).__init__()

        self.upsample = F.interpolate
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, 5)
        self.hidden_act = F.relu
        self.output_act = torch.sigmoid

    def forward(self, input):
        x = input.permute(1, -1, 0)  # rearrange to be [batch, height, width]
        x = x.unsqueeze(1)  # add singleton channel dimension

        x = self.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.hidden_act(x)

        x = self.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = self.hidden_act(x)

        x = self.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv3(x)
        x = self.output_act(x)

        return x