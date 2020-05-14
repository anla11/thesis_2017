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

from dilated_conv_block import *

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class CRNNModel(nn.Module):
    """

    """

    def __init__(self, dict_size, input_size, rnn_hidden_size):
        super(CRNNModel, self).__init__()
        self.input_size = input_size

        self.cnn1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.cnn2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.cnn3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.cnn4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.cnn5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.cnn6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.cnn7 = nn.Conv2d(512, 512, 2, 1, 0)
        self.bn3 = nn.BatchNorm2d(512)

        # Get input size
        current_input_size = 512

        self.rnn1 = nn.LSTM(input_size=current_input_size,
                            hidden_size=rnn_hidden_size,
                            bidirectional=True)
        fully_connected1 = nn.Linear(rnn_hidden_size*2, rnn_hidden_size)
        self.fc1 = SequenceWise(fully_connected1)

        self.rnn2 = nn.LSTM(input_size=rnn_hidden_size,
                            hidden_size=rnn_hidden_size,
                            bidirectional=True)

        fully_connected2 = nn.Linear(rnn_hidden_size*2, dict_size)
        self.fc2 = SequenceWise(fully_connected2)


    def forward(self, input):
        # B x F x L -> B x 1 x F x L
        input = input.unsqueeze(1)
        #print(input.size())
        input = self.cnn1(input)
        input = F.relu(input, inplace=True)
        input = self.maxpool1(input)

        input = self.cnn2(input)
        input = F.relu(input, inplace=True)
        input = self.maxpool2(input)

        input = self.cnn3(input)
        input = self.bn1(input)
        input = F.relu(input, inplace=True)

        input = self.cnn4(input)
        input = F.relu(input, inplace=True)
        input = self.maxpool3(input)

        input = self.cnn5(input)
        input = self.bn2(input)
        input = F.relu(input, inplace=True)

        input = self.cnn6(input)
        input = F.relu(input, inplace=True)
        input = self.maxpool4(input)

        input = self.cnn7(input)
        input = self.bn3(input)
        input = F.relu(input, inplace=True)

        in_sizes = input.size()
        input = input.view(in_sizes[0], in_sizes[1] * in_sizes[2], in_sizes[3])  # Collapse feature dimension
        input = input.transpose(1, 2).transpose(0, 1).contiguous()  # B x CF x L -> L x B x CF

        input, _ = self.rnn1(input)
        input = self.fc1(input)

        input, _ = self.rnn2(input)
        input = self.fc2(input)

        input = input.transpose(0, 1).contiguous()  # L x B x Dict -> B x L x Dict

        return input
