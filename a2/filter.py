f = open('test.txt', 'r')
w = open('long.txt', 'w')

for line in f:
    count = 0
    for word in line.split(' '):
        if word[0] != '(':
            count += 1
    if count > 30:
        w.write(line)
