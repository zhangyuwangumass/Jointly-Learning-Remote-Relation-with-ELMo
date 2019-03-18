a = open('small_data/abstract', 'r')
c = open('small_data/content', 'r')

s = open('small_data/sentence_75', 'w')

for l in a:
	p = c.readline()
	ab_ls = l.rstrip().split('.')
	co_ls = p.rstrip().split('.')
	for i in ab_ls:
		if i != '':
			words = i.split()
			count = 0
			for w in words:
				wl = len(w)
				if count + wl < 75:
					count += wl
					s.write(w + ' ')
				else:
					count = wl
					s.write('\n' + w)
			s.write('\n')
	for j in co_ls:
		if j != '':
			words = j.split()
			count = 0
			for w in words:
				wl = len(w)
				if count + wl < 75:
					count += wl
					s.write(w + ' ')
				else:
					count = wl
					s.write('\n' + w)
			s.write('\n')

print('Finished.')

a.close()
c.close()
s.close() 
