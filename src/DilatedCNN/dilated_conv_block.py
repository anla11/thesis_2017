'''
    @author: nguyen phuoc dat
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math


class DilatedConvBlock(nn.Module):
    """
    Accept an input tensor of size (Batch x Feature x Sequence) and output a tensor of size (Batch x Hidden x Sequence)
    """
    def __init__(self, input_size, hidden_size, kernel, dilation):
        super(DilatedConvBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size,
                               padding=dilation,
                               kernel_size=kernel,
                               dilation=dilation)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                               padding=dilation,
                               kernel_size=kernel,
                               dilation=dilation)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)

    def forward(self, input):
        # Main path
        input2 = self.batch_norm1(input)
        input2 = self.conv1(input2)
        input2 = self.batch_norm2(input2)
        input2 = F.relu(input2)
        input2 = self.conv2(input2)
        input2 = self.batch_norm3(input2)

        # Shortcut path
        #print(input.size(), input2.size())
        if self.input_size < self.hidden_size:
            input = F.pad(input.unsqueeze(1), pad=(0, 0, 0, self.hidden_size - self.input_size)).squeeze(1)
        elif self.input_size > self.hidden_size:
            input2 = F.pad(input2.unsqueeze(1), pad=(self.input_size - self.hidden_size, 0)).squeeze(1)
        #print(input.size(), input2.size())
        #Sum with shortcut path
        return input + input2