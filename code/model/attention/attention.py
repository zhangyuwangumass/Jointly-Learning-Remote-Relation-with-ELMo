import torch
import torch.nn as nn
import torch.nn.functional as F
from .lstm import LSTMUnit
import numpy as np


class AttentionDecoder(nn.Module):
    device = torch.device('cuda')
    dtype = torch.float32

    def __init__(self, input_dim, hid_dim):
        super(AttentionDecoder, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.lstm = LSTMUnit(input_dim,hid_dim).to(device=self.device, dtype=self.dtype)
        self.fc = nn.Sequential(
            nn.Linear(hid_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, input_dim)
        ).to(device=self.device, dtype=self.dtype)

    def forward(self,batch, hid_states, max_len=10):
        '''
        :param batch: input batch of shape (B x D)
        :param hid_states: recorded hidden states of shape (B x L x H)
        :return:
        '''
        B, D = batch.shape
        h = 0.1 * torch.randn(B, self.hid_dim).to(device=self.device, dtype=self.dtype)
        s = 0.1 * torch.randn(B, self.hid_dim).to(device=self.device, dtype=self.dtype)

        outputs = []

        for i in range(max_len):
            # calculate context vector
            #print('In AttentonDecoder s has shape ', s.shape, ' hid_states has shape ', hid_states.shape)
            cos_sim = (s.unsqueeze(1) * hid_states).sum(dim=2) / (((hid_states * hid_states).sum(dim=2) * (s * s).sum(dim=1,keepdim=True))**0.5 + 1e-8)
            # weight of shape (B x L)
            energy = F.softmax(cos_sim, dim=1).unsqueeze(-1)
            context = (hid_states * energy).sum(dim=1)
            s = s + context
            #print('In AttentionDecoder, input_dim is', self.input_dim, ' s has shape ', s.shape, ' h has shape ', h.shape, ' batch has shape ', batch.shape)
            h, s = self.lstm(batch, h, s)
            batch = self.fc(h)
            outputs.append(batch)

        # concatenate outputs into (B x L x D)
        L = len(outputs)
        outputs = torch.cat(outputs, dim=1).contiguous().view(B, D, L).permute(0,2,1)

        return outputs

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, hid_dim)
        )
        self.key = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, hid_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, hid_dim)
        )

    def forward(self, encoder_outputs):
        # (B, L, D) -> (B, (L x D))
        B, L, D = encoder_outputs.shape
        outputs = []
        weights = []
        for i in range(B):
            query = self.query(encoder_outputs[i])
            key = self.key(encoder_outputs[i])
            value = self.value(encoder_outputs[i])

            # scaled dot-product attention
            # (L, H) -> (L, L)
            weight = F.softmax(torch.mm(query, key.t()),dim=1)
            weights.append(weight.unsqueeze(0))
            # (L, L) * (L, H) -> (L, H)
            outputs.append(torch.mm(weight, value).unsqueeze(0))

        # final outputs and weights takes form of (B, L, H) and (B, L, L)
        outputs = torch.cat(outputs, 0)
        weights = torch.cat(weights, 0)

        return outputs, weights

class MultiHeadSelfAttention(nn.Module):

    hid_dim = None
    heads = []

    device = torch.device('cuda')
    dtype = torch.float32

    def __init__(self, input_dim, hid_dim, head_num):
        super(MultiHeadSelfAttention, self).__init__()
        self.hid_dim = hid_dim
        for i in range(head_num):
            self.heads.append(SelfAttention(input_dim, hid_dim).to(device=self.device, dtype=self.dtype))

    def forward(self, encoder_outputs):
        '''
        :param encoder_outputs: shape of (B, L, D)
        :return:
        '''
        outputs_list = []
        weights_list =[]
        for head in self.heads:
            outputs, weights = head(encoder_outputs)
            outputs_list.append(outputs)
            weights_list.append(weights)

        # Concatenate to form a B x L x (head_num x H) tensor
        outputs = torch.cat(outputs_list, 2)
        # Concatenate to form a B x L x (head_num x L) tensor
        weights = torch.cat(weights_list, 2)
        return outputs, weights

