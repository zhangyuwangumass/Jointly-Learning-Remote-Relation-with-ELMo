import torch
import numpy as np
from utils.char_vectorizer import CharVectorizer
from bilm.bilm import BiLSTMLanguageModel
from matplotlib import pyplot as plt


model = BiLSTMLanguageModel(58,16,32,3,3)
if torch.cuda.is_available(): print("CUDA is available.")

model.load_model('pretrain')

total_train_his = []
total_val_his = []

lrs = [1e-4,8e-5,7e-5,6e-5,5e-5]
thd = [(1e-4,1e-4),(7.5e-5,7.5e-5),(5e-5,5e-5),(2.5e-5,2.5e-5),(1e-5,1e-5)]

for i in range(5):
    print('partition ' + str(i) +' starts.')
    data = np.load('../data/data_tensor_' + str(i) + '.npz')['embedding']
    train_his, val_his = model.train(data,PRE=False,LR=lrs[i], MAX_EPOCH=5, NAME='partition_' + str(i), THRESHOLD=thd[i])
    np.savez('../record/partition_' + str(i) + '_history', train_his=np.array(train_his), val_his=np.array(val_his))

#np.savez('../record/history', train_his=np.array(total_train_his), val_his=np.array(total_val_his))
#EPOCH = len(total_train_his)
#plt.plot(np.arange(1,EPOCH + 1),total_train_his, label='train loss')
#plt.plot(np.arange(1,EPOCH + 1),total_val_his, label='val loss')
#plt.legend()
#plt.show()