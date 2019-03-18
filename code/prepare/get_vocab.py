
vocab = {}
o = open('../data/vocab', 'w')

for i in range(6):
	f = open('../data/sentence_' + str(i), 'r')
	for l in f:
		ws = l.rstrip().split()
		for w in ws:
			if w not in vocab:
				o.write(w + '\n')
				vocab[w] = 1
				
