f = open('../Project_Deep_Contextualized_Embeddings/prepare/content', 'r')

a = open('data/abstract', 'w')
c = open('data/content', 'w')

prev = None

count = 0
for l in f:
	if count % 2 == 1:
		if l != '\n':
			l = l.replace(',','.').replace(';', '.').replace(':','.').replace('?', '.').replace('!', '.').replace('(', '').replace(')', '').replace('[', '').replace(']', '')
			prev = prev.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
			a.write(prev)
			c.write(l)
	else:
		prev = l
	count += 1
print("Finished.")

f.close()
a.close()
c.close()
