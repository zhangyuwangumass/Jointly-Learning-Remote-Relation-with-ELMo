a = open('../full_data/abstract', 'r')
c = open('../full_data/content', 'r')

afs = []
cfs = []
for i in range(10):
	afs.append(open('../data/structured_abstract_' + str(i), 'w'))
	cfs.append(open('../data/structured_content_' + str(i), 'w'))

size = 0
MAX = 100000

while(size < MAX):
	for p in range(10):
		af = afs[p]
		cf = cfs[p]
		ab_ls = a.readline().rstrip().split('.')
		co_ls = c.readline().rstrip().split('.')

		count = 0
		for i in ab_ls:
			if i != '':
				words = i.split()
				sl = len(words)
				if count + sl < 50:
					count += sl
					af.write(i)
				else:
					count = sl
					af.write('\n' + i)
		af.write('\n\n')
		count = 0
		for j in co_ls:
			if j != '':
				words = j.split()
				sl = len(words)
				if count + sl < 50:
					count += sl
					cf.write(j)
				else:
					count = sl
					cf.write('\n' + j)
		cf.write('\n\n')
		size += 1

a.close()
c.close()
print('Finished.')

