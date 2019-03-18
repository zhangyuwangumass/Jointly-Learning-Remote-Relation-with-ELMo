import torch
import torch.nn as nn
import numpy as np
from attention.attention import MultiHeadSelfAttention
from attention.attention import AttentionDecoder
from context.context import DeepContextEncoder
from context.context import DeepContextWeighter
from torch.utils.data import DataLoader
from torch.utils.data import sampler

class Model(nn.Module):

    device = torch.device('cuda')
    dtype = torch.float32

    train_mode = False

    def __init__(self, input_dim, input_channel, max_len=10):
        super(Model,self).__init__()
        self.D = input_dim
        self.C = input_channel
        self.max_len = max_len

        self.deep_context_weighting = DeepContextWeighter(input_channel).to(device=self.device, dtype=self.dtype)
        self.sentence_attention = MultiHeadSelfAttention(input_dim, 16, 3).to(device=self.device, dtype=self.dtype)
        self.sentence_encoder = DeepContextEncoder(16 * 6, 16, 3).to(device=self.device, dtype=self.dtype)
        self.paragraph_attention = MultiHeadSelfAttention(16 * 2, 16, 3).to(device=self.device, dtype=self.dtype)
        self.paragraph_encoder = DeepContextEncoder(16 * 6, 16, 3).to(device=self.device, dtype=self.dtype)
        self.decoder = AttentionDecoder(16 * 2, 32).to(device=self.device, dtype=self.dtype)
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, input_dim)
        ).to(device=self.device, dtype=self.dtype)

    def forward(self, batch, LS=(15,1)):
        '''
        :param batch: batch is a tensor of shape (B x S x L x D x C)
        :return:
        '''
        if self.train_mode:
            B, S_M, L, D, C = batch.shape
            S, M = LS
            # (B x (S + M) x L x D x C) -> (B x (S + M) x L x D)
            # Learnable weighting on deep contextualized word embeddings
            sentence_representation = self.deep_context_weighting(batch)

            # Split sentence representation into data and truth
            # (B x (S + M) x L x D x C) -> (B x S x L x D),(B x M x L x D)
            data = sentence_representation[:, :S, :, :]
            truth = sentence_representation[:, S:, :, :]
        else:
            B, S, L, D, C = batch.shape
            # (B x S x L x D x C) -> (B x S x L x D)
            # Learnable weighting on deep contextualized word embeddings
            data = self.deep_context_weighting(batch)
            truth = None

        # Pack non-sequence length dimensions
        data = data.contiguous().view(B * S, L, D).to(device=self.device, dtype=self.dtype)

        # Run self-attention to encode intra-sentence relation into vectors
        # ((B x S) x L x D) -> ((B x S) x L x H)
        sen_att, sen_att_weight = self.sentence_attention(data)
        #print('In DeepAttention model, sentence attention has shape ', sen_att.shape)

        # Feed sentence representation into the deep contextual encoder
        # to get the sentence embedding
        # ((B x S) x L x H) -> ((B x S) x L x E), ((B x S) x E)
        sen_states, sen_embedding = self.sentence_encoder(sen_att)
        #print('In DeepAttention model, sentence states has shape ', sen_states.shape)

        # Unpack dimensions
        sen_embedding = sen_embedding.view(B, S, -1).to(device=self.device, dtype=self.dtype)
        #print('In DeepAttention model, sentence embedding has shape ', sen_embedding.shape)

        # Run self-attention to encode inter-sentence relation into vectors
        # (B x S x E) -> (B x S x H')
        para_att, para_att_weight = self.paragraph_attention(sen_embedding)
        #print('In DeepAttention model, paragraph attention has shape ', para_att.shape)

        # Use paragraph attention tensor to encode a paragraph representation
        # (B x S x H') -> (B x S x H'), (B x H')
        para_states, para_embedding = self.paragraph_encoder(para_att)
        #print('In DeepAttention model, paragraph states has shape ', para_states.shape)
        # Decoder outputs as (B x max_len x H')
        outputs = self.decoder(para_embedding, para_states)
        #print('In DeepAttention model, preliminary outputs has shape ', outputs.shape)
        # FC to transform outputs into the same dimension as the input word embedding (B x max_len x input_dim)
        return self.fc(outputs), truth

    def context_weighting(self, x):
        return self.deep_context_weighting(x)

class DeepAttentionModel:

    device = torch.device('cuda')
    dtype = torch.float32

    pretrain_threshold = 0.005
    prevalid_threshold = 0.006

    def __init__(self, input_dim, input_channel, max_len=10):
        self.model = Model(input_dim, input_channel, max_len=max_len).to(device=self.device, dtype=self.dtype)

    def loss(self, x, y):
        '''
        :param x: predicted sequence with shape of (B x max_len x D)
        :param y: ground truth sequence with shape of (B x M x L x D)
        :return:
        '''
        B = x.shape[0]
        # average the vectors within a sentence into a combined vector
        #print('x original shape is ', x.shape)
        #print('y original shape is ', y.shape)
        y = y.mean(dim=2).squeeze(2)
        #print('x shape is ', x.shape)
        #print('y shape is ', y.shape)
        sum = 0
        for i in range(B):
            prod = torch.mm(x[i], y[i].t())
            # scale over the length of x and y
            len_x = ((x[i] * x[i]).sum(dim=1, keepdim=True))**0.5
            len_y = (((y[i] * y[i]).sum(dim=1, keepdim=True))**0.5).transpose(1,0)
            prod /= len_x
            prod /= len_y
            sum += (1 - prod).mean()

        #naive loss: averaging
        return sum / B

    def train(self, data, LS=(15,1), BATCH=64, VAL=0.1, LR=1e-4, MAX_EPOCH=10, PRE=False, NAME='checkpoint', THRESHOLD=(0.001, 0.001)):

        '''
        :param data: data is of shape (N x (LP + LA) x D)
        :param LS: tuple indicating how long the data vector is and how long the ground truth vector is
        :param BATCH:
        :param VAL:
        :param LR:
        :param MAX_EPOCH:
        :param PRE:
        :param NAME:
        :param THRESHOLD:
        :return:
        '''

        # set model at training mode]
        self.model.train_mode = True

        if PRE:
            train_threshold = self.pretrain_threshold
            valid_threshold = self.prevalid_threshold
            name = NAME + '_pretrain'
        else:
            train_threshold, valid_threshold = THRESHOLD
            name = NAME + '_' + str(LR)

        NUM = data.shape[0]
        VAL = int(NUM * VAL)
        loader_train = DataLoader(data, batch_size=BATCH,
                                  sampler=sampler.SubsetRandomSampler(range(NUM - VAL)))
        loader_valid = DataLoader(data, batch_size=BATCH,
                                  sampler=sampler.SubsetRandomSampler(range(NUM - VAL, NUM)))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, betas=(0.5, 0.999))

        train_loss_history = []
        valid_loss_history = []

        for e in range(MAX_EPOCH):
            print("train epoch " + str(e) + " starts")
            count = 0
            train_loss = 0
            valid_loss = 0
            for i, batch in enumerate(loader_train):

                self.model.train()

                #data = batch[:,:P,:]
                #truth = batch[:,P:,:]
                #print('data shape is ', data.shape)
                #print('truth shape is ', truth.shape)

                #data = data.to(device=self.device, dtype=self.dtype)
                #truth = truth.to(device=self.device, dtype=self.dtype)
                batch = batch.to(device=self.device, dtype=self.dtype)

                recon, truth = self.model(batch)
                loss = self.loss(recon, truth)
                train_loss += loss.item()
                train_loss_history.append(loss.item())
                count += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #if i % 100 == 0:
                print('Iteration %d, train loss = %.4f' % (i, loss.item()))
            train_loss /= count
            count = 0
            for i, batch in enumerate(loader_valid):

                self.model.eval()
                #data = batch[:, :P, :]
                #truth = batch[:, P:, :]

                with torch.no_grad():
                    #data = data.to(device=self.device, dtype=self.dtype)
                    #truth = truth.to(device=self.device, dtype=self.dtype)
                    batch = batch.to(device=self.device, dtype=self.dtype)

                    recon, truth = self.model(batch)
                    loss = self.loss(recon, truth)
                    valid_loss += loss.item()
                    valid_loss_history.append(loss.item())
                    count += 1

                #if i % 100 == 0:
                print('Iteration %d, validation loss = %.4f' % (i, loss.item()))

            valid_loss /= count

            #train_loss_history.append(train_loss)
            #valid_loss_history.append(valid_loss)

            print("train_loss ", train_loss)
            print("valid_loss ", valid_loss)
            self.save_model(name + '_epoch_' + str(e))
            if (train_loss < train_threshold and valid_loss < valid_threshold): break
        return train_loss_history, valid_loss_history

    def predict(self, batch):
        # set model at predict mode
        self.model.train_mode = False
        self.model.eval()
        with torch.no_grad():
            batch = batch.to(device=self.device, dtype=self.dtype)
            recon, truth = self.model(batch)
        # predicted sequence (B x max_len x input_dim)
        return recon.cpu().detach().numpy()

    def embed(self, vocab_matrix):
        # vocab matrix of shape (N x (depth x dir) x H)
        vocab_matrix = np.load(vocab_matrix)['embedding']
        N, dp_dir, H = vocab_matrix.shape
        print(vocab_matrix.shape)
        # (N x (depth x dir) x H) -> (N x depth x (dir x H)) -> (N x (dir x H) x depth)
        vocab_matrix = torch.from_numpy(vocab_matrix.reshape(N, 3, -1).transpose((0,2,1))).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            # weighting vocab_matrix to get a (N x (dir x H)) embedding matrix
            np.savez('../data/vocab_embedding', embedding=self.model.context_weighting(vocab_matrix).cpu().detach().numpy())


    def save_model(self, name):
        """Save the model to disk

        Arguments: none
        """
        torch.save(self.model.state_dict(), '../checkpoint/' + name + '.pkl')

    def load_model(self, name):
        """Load the saved model from disk

        Arguments: none
        """
        self.model.load_state_dict(torch.load('../checkpoint/' + name + '.pkl'))
