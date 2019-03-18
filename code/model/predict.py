import torch
import numpy as np
from utils.char_vectorizer import CharVectorizer
from utils.word_from_embedding import EmbeddingGenerator
from deep_attention import DeepAttentionModel


model = DeepAttentionModel(32, 3, max_len=10)
if torch.cuda.is_available(): print("CUDA is available.")

model.load_model('matrix_0_pretrain')

model.embed('../data/vocab_matrix.npz')

data = np.load('../data/data_matrix_0.npz')['embedding']
paragraph = torch.from_numpy(data[20,:15]).unsqueeze(0)
print(paragraph.shape)

# prediction is of shape (B x max_len x 2H)
pred = model.predict(paragraph)
generator = EmbeddingGenerator('../data/vocab', '../data/vocab_embedding')

B, max_len, H = pred.shape

for i in range(B):
    for j in range(max_len):
        generator.generate_with_best(pred[i,j,:].reshape(1,H))



#total_train_his = []
#total_val_his = []

#lrs = [1e-4,8e-5,7e-5,6e-5,5e-5]
#thd = [(1e-4,1e-4),(7.5e-5,7.5e-5),(5e-5,5e-5),(2.5e-5,2.5e-5),(1e-5,1e-5)]

#for i in range(5):
    #print('partition ' + str(i) +' starts.')
    #data = np.load('../data/data_tensor_' + str(i) + '.npz')['embedding']
    #train_his, val_his = model.train(data,PRE=False,LR=lrs[i], MAX_EPOCH=5, NAME='partition_' + str(i), THRESHOLD=thd[i])
    #np.savez('../record/partition_' + str(i) + '_history', train_his=np.array(train_his), val_his=np.array(val_his))
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