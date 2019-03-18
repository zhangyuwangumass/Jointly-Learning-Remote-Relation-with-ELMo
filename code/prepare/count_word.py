import sys

f = open(sys.argv[1], 'r')
max_len = int(sys.argv[2])
print(max_len)

length = 0
Count = 0
sen = ''

for l in f:
	ltemp = len(l.rstrip().split())
	if ltemp > length:
		length = ltemp
		sen = l
	if ltemp > max_len:
		Count += 1
		print(ltemp)
		print(l)

print(length)
print(sen)
print(Count)
