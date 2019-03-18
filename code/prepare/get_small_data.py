a = open('../full_data/abstract', 'r')
c = open('../full_data/content', 'r')

fs = []
for i in range(10):
	fs.append(open('../data/sentence_' + str(i), 'w'))

size = 0
MAX = 100000

while(size < MAX):
	for f in fs:
		ab_ls = a.readline().rstrip().split('.')
		co_ls = c.readline().rstrip().split('.')

		for i in ab_ls:
			if i != '':
				words = i.split()
				count = 0
				for w in words:
					wl = len(w)
					if count + wl < 75:
						count += wl
						f.write(w + ' ')
					else:
						count = wl
						f.write('\n' + w)
				f.write('\n')
		for j in co_ls:
			if j != '':
				words = j.split()
				count = 0
				for w in words:
					wl = len(w)
					if count + wl < 75:
						count += wl
						f.write(w + ' ')
					else:
						count = wl
						f.write('\n' + w)
				f.write('\n')
		size += 1

a.close()
c.close()
print('Finished.')
