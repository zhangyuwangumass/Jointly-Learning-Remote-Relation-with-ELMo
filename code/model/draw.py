from matplotlib import pyplot as plt
import numpy as np

train_his = []
val_his = []
for i in range(1,5):
    #train_his += np.load('../record/matrix_' + str(i) +'_history.npz')['train_his'].tolist()
    val_his += np.load('../record/matrix_' + str(i) +'_history.npz')['val_his'].tolist()

#EPOCH = len(train_his)
EPOCH = len(val_his)

#plt.plot(np.arange(1,EPOCH + 1), train_his, label='train loss')
plt.plot(np.arange(1,EPOCH + 1), val_his, label='val loss')

plt.legend()

plt.show()