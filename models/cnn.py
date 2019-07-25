import numpy as np
from scipy import interpolate

import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(1, 6, 3, padding=2)
        self.conv2 = nn.Conv2d(6, 6, 3, padding=2)
        self.conv3 = nn.Conv2d(6, 1, 3, padding=1)
        self.hidden_act = F.relu
        self.output_act = torch.sigmoid

    def forward(self, input):
        x = self.conv1(input)
        x = self.pool(x)
        x = self.hidden_act(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.hidden_act(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = self.output_act(x)

        return x