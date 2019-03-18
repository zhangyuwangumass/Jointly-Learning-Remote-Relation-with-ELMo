import torch
import numpy as np
from utils.char_vectorizer import CharVectorizer
from deep_attention import DeepAttentionModel


model = DeepAttentionModel(32, 3, max_len=10)
if torch.cuda.is_available(): print("CUDA is available.")

model.load_model('matrix_4_addon_5e-05_epoch_1')

#total_train_his = []
#total_val_his = []

#lrs = [0,8e-5,7e-5,6e-5,5e-5]
#thd = [(0,0),(7.5e-5,7.5e-5),(5e-5,5e-5),(2.5e-5,2.5e-5),(1e-5,1e-5)]

lrs = [5e-5,4e-5,3e-5,2e-5,1e-5]
thd = [(1e-5,1e-5),(8e-6,8e-6),(6e-6,6e-6),(4e-6,4e-6),(2e-6,2e-6)]


for i in range(5):
    print('partition ' + str(i) +' starts.')
    data = np.load('../data/data_matrix_' + str(i) + '.npz')['embedding']
    train_his, val_his = model.train(data,PRE=False,LR=lrs[i], MAX_EPOCH=3, NAME='fine_training_' + str(i), THRESHOLD=thd[i])
    #np.savez('../record/partition_' + str(i) + '_history', train_his=np.array(train_his), val_his=np.array(val_his))
    np.savez('../record/fine_training_' + str(i) + '_history', train_his=np.array(train_his), val_his=np.array(val_his))
'''
data = np.load('../data/data_matrix_4.npz')['embedding']
train_his, val_his = model.train(data,PRE=False,LR=5e-5, MAX_EPOCH=3, NAME='matrix_4_addon', THRESHOLD=(1e-5,1e-5))

#torch.backends.cudnn.enabled=False

#data = np.load('../data/data_matrix_0.npz')['embedding']
#print(data.shape)
#train_his, val_his = model.train(data,PRE=True,BATCH=32, LR=1e-4, MAX_EPOCH=5, NAME='matrix_0')
#np.savez('../record/matrix_0_history', train_his=np.array(train_his), val_his=np.array(val_his))

#np.savez('../record/history', train_his=np.array(total_train_his), val_his=np.array(total_val_his))
#EPOCH = len(total_train_his)
#plt.plot(np.arange(1,EPOCH + 1),total_train_his, label='train loss')
#plt.plot(np.arange(1,EPOCH + 1),total_val_his, label='val loss')
#plt.legend()
#plt.show()
'''