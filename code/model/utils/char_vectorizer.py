import numpy as np
import torch
import torch.nn as nn


class CharVectorizer:

    feature_dim = None
    char_index = {}

    def __init__(self, extra_char=(':', '\'')):
        for i in range(26):
            self.char_index[chr(ord('A') + i)] = 3 + i

        for i in range(26):
            self.char_index[chr(ord('a') + i)] = 29 + i

        self.char_index[' '] = 55

        if extra_char != None:
            for j in range(len(extra_char)):
                self.char_index[extra_char[j]] = 56 + j

        #print(self.char_index)

        self.feature_dim = 56 + len(extra_char)
        '''
        The first 3 dimensions are reserved for start(<S>), end(</S>) and unknown(</N>)
        '''

    def vectorize(self, sequence, max_len):
        '''
        :param sequence: original list of N strings
        :return:
        '''
        N = len(sequence)
        data = np.zeros((N, max_len + 2, self.feature_dim))
        data[:,:,0] += 1
        for i in range(N):
            l = sequence[i].rstrip()
            L = min(len(l), max_len)
            for j in range(L):
                c = l[j]
                if c in self.char_index:
                    data[i,j + 1,self.char_index[c]] = 1
                else:
                    data[i, j + 1, 2] = 1

            for k in range(L + 1, max_len + 2):
                data[i, k, 1] = 1

        #np.savetxt('../data/data_tensor.npy', data)
        return data

    def vocab_vec_and_save(self, sequence, max_len):
        N = len(sequence)
        data = np.zeros((N + 3, max_len + 2, self.feature_dim))
        # set <Start>
        data[0, 1, 0] += 1
        data[0, 2:,1] += 1
        # set <End>
        data[1, 1:,1] += 1
        # set <Unknown>
        data[2, 1, 2] += 1
        data[2, 2:,1] += 1
        # set start for all words
        data[:, 0, 0] += 1
        for i in range(N):
            l = sequence[i].rstrip()
            L = min(len(l), max_len)
            for j in range(L):
                c = l[j]
                if c in self.char_index:
                    data[i + 3, j + 1, self.char_index[c]] = 1
                else:
                    data[i + 3, j + 1, 2] = 1

            for k in range(L + 1, max_len + 2):
                data[i + 3, k, 1] = 1

        # np.savetxt('../data/data_tensor.npy', data)
        np.savez('../data/vocab_tensor', embedding=data)

    def vec_and_save(self, sequence, max_len, name):
        np.savez('../data/data_tensor_' + name, embedding=self.vectorize(sequence, max_len))

