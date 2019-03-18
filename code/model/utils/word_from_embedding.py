import numpy as np
import torch
import torch.nn as nn


class EmbeddingGenerator:

    def __init__(self, vocab, vocab_embedding):
        # vocab list
        self.vocab = open(vocab, 'r').readlines()
        # vocab embedding matrix of shape (N x 2H)
        self.vocab_embedding = np.load(vocab_embedding + '.npz')['embedding']

    def cosine_similarity(self, x):
        # x is of shape (1 x 2H)
        prod = (self.vocab_embedding * x).sum(axis=1)
        len_vocab = ((self.vocab_embedding * self.vocab_embedding).sum(axis=1)) ** 0.5
        len_x = ((x * x).sum()) ** 0.5
        sim = prod / (len_vocab * len_x)
        return sim

    def generate_within_delta(self, x, delta):
        # delta is the difference from being identical
        sim = self.cosine_similarity(x)
        mask = sim[sim > 1 - delta]
        for m in mask:
            print(self.vocab[m])

    def generate_with_capacity(self, x, cap):
        # cap is the size of words we include
        sim = self.cosine_similarity(x)
        pass

    def generate_with_best(self, x):
        sim = self.cosine_similarity(x)
        m = np.argmax(sim)
        #print(m)
        #print(self.vocab[m])
        print(sim)
        print()


