import json
import numpy as np
import torch
import torch.nn as nn
from .bilm import BiLSTMLanguageModel

class ELMoModel:

    model = None

    vocab_matrix = None

    dict = None

    device = torch.device('cuda')
    dtype = torch.float32

    bidirec = None
    direc = None

    en_depth = None
    de_depth = None

    input_dim = None
    en_hid_dim = None
    de_hid_dim = None

    def load_option(self, option):
        option = json.load(open(option, 'r'))
        self.input_dim = option['input_dim']
        self.en_hid_dim = option['en_hid_dim']
        self.de_hid_dim = option['de_hid_dim']
        self.en_depth = option['en_depth']
        self.de_depth = option['de_depth']
        self.bidirec = option['bidirec']
        if self.bidirec:
            self.direc = 2
        else:
            self.direc = 1
        print('direc is ', self.direc)
        self.model = BiLSTMLanguageModel(self.input_dim, self.en_hid_dim, self.de_hid_dim, self.en_depth, self.de_depth, bidirec=self.bidirec)

    def load_model(self, name):
        self.model.load_model(name)

    def load_voc_matrix(self, name):
        self.vocab_matrix = np.load('../data/' + name + '.npz')['embedding']

    def embed_vocab(self, vocab_tensor, BATCH=64):
        vocab_tensor = torch.from_numpy(np.load(vocab_tensor)['embedding']).to(device=self.device, dtype=self.dtype)
        N, L, D = vocab_tensor.shape
        self.vocab_matrix = np.zeros((self.direc * self.en_depth, N, self.en_hid_dim))
        print('initial vocab matrix shape is ', self.vocab_matrix.shape)
        prev = 0
        for i in range(N):
            #print('embedding batch ' + str(i))
            if i % BATCH == 0 and i > 0:
                self.vocab_matrix[:,prev:i,:] = self.model.encode(vocab_tensor[prev:i], BATCH).cpu().numpy()
                prev = i
        self.vocab_matrix[:, prev:N, :] = self.model.encode(vocab_tensor[prev:N], N-prev).cpu().numpy()
        self.vocab_matrix = self.vocab_matrix.transpose((1,0,2))
        print('final vocab matrix shape is ', self.vocab_matrix.shape)
        np.savez('../data/vocab_matrix', embedding=self.vocab_matrix)

    def embed_data(self, data, truth, name):
        '''
        Embedding a batch of paragraphs into embedding.
        :param data: tensor of shape (P x S x L), where each point takes an int value, indicating the word index at that position
        :param truth: tensor of shape (P x M x L), where each point takes an int value, indicating the word index at that position
        :return: tensor of shape ((P x (S + M)) x L x (D x dir) x depth)
        '''
        P, S, L = data.shape
        P, M, L = truth.shape
        print(self.vocab_matrix.shape)
        data_matrix = self.vocab_matrix[data,:,:]
        truth_matrix = self.vocab_matrix[truth,:,:]
        print('data matrix shape is ', data_matrix.shape)
        print('truth matrix shape is ', truth_matrix.shape)
        matrix = np.concatenate((data_matrix, truth_matrix), axis=1)
        print(matrix.shape)
        matrix = matrix.reshape(P, (S + M), L, self.en_depth, -1).transpose((0,1,2,4,3))
        print(matrix.shape)
        np.savez('../data/' + name, embedding=matrix)






