import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler

class LSTMSqueeze(nn.Module):
    def forward(self, x):
        '''
        :param x: a tensor of shape (B x L x H), produced by a LSTM
        :return: a tensor of shape ((B x L) x H)
        '''
        #print(x.shape)
        B, L, H = x.shape
        return x.contiguous().view(-1, H)

class BiLSTMLanguageModel:

    device = torch.device('cuda')
    dtype = torch.float32

    direc = None

    en_depth = None
    de_depth = None

    input_dim = None
    en_hid_dim = None
    de_hid_dim = None

    encoder = None
    decoder = None
    fc = None

    pretrain_threshold = 0.005
    prevalid_threshold = 0.006

    def __init__(self, input_dim, en_hid_dim, de_hid_dim, en_depth, de_depth, bidirec=True):
        '''
        :param input_dim: input data feature dim, int
        :param hidden_dim: hidden layer dim, int
        :param depth:
        '''
        if bidirec:
            self.direc = 2
        else:
            self.direc = 1

        self.en_depth = en_depth
        self.de_depth = de_depth

        self.input_dim = input_dim
        self.en_hid_dim = en_hid_dim
        self.de_hid_dim = de_hid_dim

        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=en_hid_dim,
                               num_layers=en_depth, batch_first=True,
                               bidirectional=bidirec).to(device=self.device, dtype=self.dtype)
        self.decoder = nn.LSTM(input_size=self.direc * en_hid_dim, hidden_size=de_hid_dim,
                               num_layers=de_depth, batch_first=True,
                               bidirectional=bidirec).to(device=self.device, dtype=self.dtype)
        self.fc = nn.Sequential(
            LSTMSqueeze(),
            nn.Linear(self.direc * de_hid_dim, input_dim),
            nn.Sigmoid()
        ).to(device=self.device, dtype=self.dtype)

    def forward(self, batch, batch_size):
        '''
        :param data: Tensor of shape (L x B x D)
        :return:
        '''
        #print(batch.shape)
        #print(batch_size)
        he0 = torch.randn(self.en_depth * self.direc, batch_size, self.en_hid_dim).to(device=self.device, dtype=self.dtype)
        ce0 = torch.randn(self.en_depth * self.direc, batch_size, self.en_hid_dim).to(device=self.device, dtype=self.dtype)
        #print(he0.shape)
        #print(ce0.shape)
        code, _ = self.encoder(batch, (he0, ce0))
        #print(code.shape)

        hd0 = torch.randn(self.de_depth * self.direc, batch_size, self.de_hid_dim).to(device=self.device, dtype=self.dtype)
        cd0 = torch.randn(self.de_depth * self.direc, batch_size, self.de_hid_dim).to(device=self.device, dtype=self.dtype)
        #print(hd0.shape)
        #print(cd0.shape)
        out, _ = self.decoder(code, (hd0, cd0))
        #print(out.shape)

        return self.fc(out)

    def encode(self, batch, batch_size):
        self.encoder.eval()
        with torch.no_grad():
            he0 = torch.randn(self.en_depth * self.direc, batch_size, self.en_hid_dim).to(device=self.device,
                                                                                      dtype=self.dtype)
            ce0 = torch.randn(self.en_depth * self.direc, batch_size, self.en_hid_dim).to(device=self.device,
                                                                                      dtype=self.dtype)
            code, (h_n, c_n) = self.encoder(batch, (he0, ce0))

        return h_n

    def train(self, data, BATCH=64, VAL=0.1, LR=1e-4, MAX_EPOCH=10, PRE=False, NAME='checkpoint', THRESHOLD=(0.001, 0.001)):

        if PRE:
            train_threshold = self.pretrain_threshold
            valid_threshold = self.prevalid_threshold
            name = NAME + '_pretrain'
        else:
            train_threshold, valid_threshold = THRESHOLD
            name = NAME + '_' + str(LR)

        NUM, LEN, DIM = data.shape
        VAL = int(NUM * VAL)
        loader_train = DataLoader(data, batch_size=BATCH,
                                  sampler=sampler.SubsetRandomSampler(range(NUM - VAL)))
        loader_valid = DataLoader(data, batch_size=BATCH,
                                  sampler=sampler.SubsetRandomSampler(range(NUM - VAL, NUM)))
        optimizer = torch.optim.Adam([
                {'params': self.encoder.parameters()},
                {'params': self.decoder.parameters()},
                {'params': self.fc.parameters(), 'lr': 1e-4}
            ], lr=LR, betas=(0.5, 0.999))

        train_loss_history = []
        valid_loss_history = []

        for e in range(MAX_EPOCH):
            print("train epoch " + str(e) + " starts")
            count = 0
            train_loss = 0
            valid_loss = 0
            for i, batch in enumerate(loader_train):

                self.encoder.train()
                self.decoder.train()
                self.fc.train()

                N = batch.shape[0]
                batch = batch.to(device=self.device, dtype=self.dtype)

                batch_recon = self.forward(batch, N)
                loss = torch.nn.functional.mse_loss(batch_recon, batch.contiguous().view(-1, DIM))
                train_loss += loss.item()
                count += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print('Iteration %d, train loss = %.4f' % (i, loss.item()))
            train_loss /= count
            count = 0
            for i, batch in enumerate(loader_valid):

                self.encoder.eval()
                self.decoder.eval()
                self.fc.eval()

                with torch.no_grad():
                    N = batch.shape[0]
                    batch = batch.to(device=self.device, dtype=self.dtype)

                    batch_recon = self.forward(batch, N)
                    loss = torch.nn.functional.mse_loss(batch_recon, batch.contiguous().view(-1, DIM))
                    valid_loss += loss.item()
                    count += 1

                if i % 100 == 0:
                    print('Iteration %d, validation loss = %.4f' % (i, loss.item()))

            valid_loss /= count

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)

            print("train_loss ", train_loss)
            print("valid_loss ", valid_loss)
            if (train_loss < train_threshold and valid_loss < valid_threshold): break

        self.save_model(name)
        return train_loss_history, valid_loss_history

    def batch_train(self, batch, LR=1e-4):

        self.encoder.train()
        self.decoder.train()
        self.fc.train()

        NUM, LEN, DIM = batch.shape

        optimizer = torch.optim.Adam([
                {'params': self.encoder.parameters()},
                {'params': self.decoder.parameters()},
                {'params': self.fc.parameters(), 'lr': 1e-4}
            ], lr=LR, betas=(0.5, 0.999))

        train_loss = 0

        batch = torch.from_numpy(batch).to(device=self.device, dtype=self.dtype)

        batch_recon = self.forward(batch, NUM)
        loss = torch.nn.functional.mse_loss(batch_recon, batch.contiguous().view(-1, DIM))
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
        
        #print('Train loss = %.4f' % loss.item())

    def save_model(self, name):
        """Save the model to disk

        Arguments: none
        """
        torch.save(self.encoder.state_dict(), '../checkpoint/' + name + '_encoder.pkl')
        torch.save(self.decoder.state_dict(), '../checkpoint/' + name + '_decoder.pkl')
        torch.save(self.fc.state_dict(), '../checkpoint/' + name + '_fc.pkl')

    def load_model(self, name):
        """Load the saved model from disk

        Arguments: none
        """
        self.encoder.load_state_dict(torch.load('../checkpoint/' + name + '_encoder.pkl'))
        self.decoder.load_state_dict(torch.load('../checkpoint/' + name + '_decoder.pkl'))
        self.fc.load_state_dict(torch.load('../checkpoint/' + name + '_fc.pkl'))