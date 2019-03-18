from bilm.elmo import ELMoModel
import numpy as np

elmo = ELMoModel()
elmo.load_option('../checkpoint/option.json')
#elmo.load_model('partition_4_5e-05')
#elmo.embed_vocab('../data/vocab_tensor.npz')


elmo.load_voc_matrix('vocab_matrix')
for i in range(0,2):
    abs = np.load('../data/abstract_index_' + str(i) + '.npz')['index']
    con = np.load('../data/content_index_' + str(i) + '.npz')['index']
    abs.dtype = np.int32
    con.dtype = np.int32

    print(np.max(abs))
    print(np.max(con))

    elmo.embed_data(abs, con, 'data_matrix_' + str(i))
