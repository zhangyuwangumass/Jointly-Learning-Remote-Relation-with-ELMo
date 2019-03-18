import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMSplit(nn.Module):
    def forward(self, x):
        return torch.chunk(x, 4, dim=-1)

class LSTMUnit(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(LSTMUnit, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.x_projection = nn.Linear(input_dim, 4 * hid_dim)
        self.h_projection = nn.Linear(hid_dim, 4 * hid_dim)
        self.split = LSTMSplit()

    def forward(self, x, h, s):
        a = self.x_projection(x) + self.h_projection(h)
        ai, af, ao, ag = self.split(a)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = F.tanh(ag)
        next_s = f * s + i * g
        next_h = o * F.tanh(next_s)
        return next_h, next_s


