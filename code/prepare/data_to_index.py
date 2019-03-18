import numpy as np

vocab_file = open('../data/vocab', 'r')
vocab_map = {}

vocab_map['<S>'] = 0
vocab_map['</S>'] = 1
vocab_map['<Unk>'] = 2

count = 3
for l in vocab_file:
	l = l.rstrip()
	vocab_map[l] = count
	count += 1

print(len(vocab_map.keys()))


for i in range(10):
	a = open('../data/structured_abstract_' + str(i), 'r')
	c = open('../data/structured_content_' + str(i), 'r')

	als = a.readlines()
	cls = c.readlines()

	N = 10000
	S = 15
	M = 1
	L = 50

	abs_mat = np.zeros((N, M, L + 2), dtype=np.int32)
	con_mat = np.zeros((N, S, L + 2), dtype=np.int32)

	abs_mat[:,:,0] = 0
	abs_mat[:,:,-1] = 1

	con_mat[:, :, 0] = 0
	con_mat[:, :, -1] = 1

	als_cur = 0
	cls_cur = 0
	for j in range(N):
		s_n = 0
		while(als[als_cur] != '\n'):
			if s_n < M:
				ws = als[als_cur].rstrip().split()
				max_l = min(L, len(ws))
				for k in range(max_l):
					if ws[k] in vocab_map:
						abs_mat[j, s_n, k + 1] = vocab_map[ws[k]]
					else:
						abs_mat[j, s_n, k + 1] = 2

				if max_l < L:
					for q in range(max_l + 1, L + 1):
						abs_mat[j, s_n, q] = 1
			s_n += 1
			als_cur += 1
		als_cur += 1

		s_n = 0
		while (cls[cls_cur] != '\n'):
			if s_n < S:
				ws = cls[cls_cur].rstrip().split()
				max_l = min(L, len(ws))
				for k in range(max_l):
					if ws[k] in vocab_map:
						con_mat[j, s_n, k + 1] = vocab_map[ws[k]]
					else:
						con_mat[j, s_n, k + 1] = 2

				if max_l < L:
					for q in range(max_l + 1, L + 1):
						con_mat[j, s_n, q] = 1
			s_n += 1
			cls_cur += 1
		cls_cur += 1

	np.savez('../data/abstract_index_' + str(i), index=abs_mat)
	np.savez('../data/content_index_' + str(i), index=con_mat)

	a.close()
	c.close()
print('Finished.')

