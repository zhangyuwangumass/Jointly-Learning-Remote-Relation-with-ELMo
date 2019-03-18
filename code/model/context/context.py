import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler

class DeepContextWeighter(nn.Module):
    def __init__(self,input_channel):
        super(DeepContextWeighter,self).__init__()
        self.input_channel = input_channel
        self.projection = nn.Linear(input_channel, 1)

    def forward(self, x):
        # x is of shape (N, *, input_channel)
        return self.projection(x).squeeze(-1)


class DeepContextEncoder(nn.Module):
    device = torch.device('cuda')
    dtype = torch.float32

    def __init__(self, input_dim, hid_dim, depth, bidirec=True):
        super(DeepContextEncoder, self).__init__()
        self.depth = depth
        self.hid_dim = hid_dim
        if bidirec:
            self.direc = 2
        else:
            self.direc = 1
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hid_dim,
                               num_layers=depth, batch_first=True,
                               bidirectional=bidirec).to(device=self.device, dtype=self.dtype)
        self.weighter = DeepContextWeighter(depth).to(device=self.device, dtype=self.dtype)

    def forward(self, batch):
        '''
        :param batch: (B x L x D)
        :return: (B x 2H)
        '''
        batch_size = batch.shape[0]
        he0 = torch.randn(self.depth * self.direc, batch_size, self.hid_dim).to(device=self.device, dtype=self.dtype)
        ce0 = torch.randn(self.depth * self.direc, batch_size, self.hid_dim).to(device=self.device, dtype=self.dtype)

        #print('In DeepContextEncoder, batch shape is ', batch.shape)
        #print('In DeepContextEncoder, he0 shape is ', he0.shape)
        #print('In DeepContextEncoder, ce0 shape is ', ce0.shape)

        code, (h_n, c_n) = self.encoder(batch, (he0, ce0))
        #print('In DeepContextEncoder, h_n shape is ', h_n.shape)
        #print('depth, direc, batch size are ', self.depth, ' ', self.direc, ' ', batch_size)
        h_n = h_n.view(self.depth, self.direc, batch_size, self.hid_dim).permute(2,1,3,0).contiguous().view(batch_size, self.direc * self.hid_dim, self.depth).to(device=self.device, dtype=self.dtype)
        # return shape a hidden_state of (B x L x 2H) and a weighted sum of embedding of (B x 2H)
        return code, self.weighter(h_n)