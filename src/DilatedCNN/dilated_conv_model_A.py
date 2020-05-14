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


class DilatedConvModelA(nn.Module):
    """
    
    """

    def __init__(self, dict_size, input_size, grow_init, kernel, grow_rate, max_dilation):
        super(DilatedConvModelA, self).__init__()
        self.input_size = input_size
        self.super_block_1 = nn.ModuleList()
        current_dilation = 1
        current_input_size = input_size
        current_hidden_size = input_size + grow_init
        while current_dilation <= max_dilation:
            self.super_block_1.append(DilatedConvBlock(current_input_size,
                                                       current_hidden_size,
                                                       kernel,
                                                       current_dilation))
            current_input_size = current_hidden_size
            current_hidden_size += grow_rate
            current_dilation *= 2

        # self.downsampler1 = nn.Conv1d(current_input_size, current_input_size * 2, 1, 2)
        #
        # self.super_block_2 = nn.ModuleList()
        # current_dilation = 1
        # current_input_size = current_input_size * 2
        # current_hidden_size = current_input_size + grow_init
        # while current_dilation <= max_dilation:
        #     self.super_block_2.append(DilatedConvBlock(current_input_size,
        #                                                current_hidden_size,
        #                                                kernel,
        #                                                current_dilation))
        #     current_input_size = current_hidden_size
        #     current_hidden_size = current_hidden_size + grow_rate
        #     current_dilation *= 2

        self.output_size = current_hidden_size
        self.avg_pool = nn.AdaptiveAvgPool1d(dict_size)

    def forward(self, input):
        # B x L X F -> B x F x L
        #input = input.tranpose(1, 2).contiguous()
        # First group
        for block in self.super_block_1:
            input = block(input)

        # # Reduce length
        # input = self.downsampler1(input)
        #
        # # Second group
        # for block in self.super_block_2:
        #     input = block(input)

        # B x F X L -> B x L x F
        input = input.transpose(1, 2).contiguous()
        input = self.avg_pool(input)

        return input
