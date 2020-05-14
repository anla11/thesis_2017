import torch
from torch.autograd import Variable
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, nIn, nHidden, nLayer, nOut):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_size = nIn, hidden_size = nHidden, num_layers = nLayer)
        self.linear = nn.Linear(nHidden, nOut)
    
    def forward(self, input, state0): #state0 = (h0, c0)
        output, _ = self.rnn(input, state0)
        T, b, h = output.size()
        output = output.view(T * b, h)        
        output = self.linear(output)
        output = output.view(T, b, -1)
        return output        

