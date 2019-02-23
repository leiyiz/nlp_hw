import numpy as np
from scipy import spatial as sp
import sys

if len(sys.argv) != 2:
    print('usage: python top10.py "the word you want to use"')
    sys.exit(1)

f = open('/cse/web/courses/cse447/19wi/assignments/resources/glove-uncompressed/glove.6B.300d.txt', 'r')

record = {}

for line in f:
    units = line.split(' ')
    word = units.pop(0)
    units = list(map(float, units))
    word_vec = np.array(units)
    record[word] = word_vec
    pass

target_word = sys.argv[1]
target_vec = record[target_word]

vec = []
for word in record:
    temp_vec = record[word]
    cos_dis = sp.distance.cosine(target_vec, temp_vec)
    vec.append((cos_dis, word))
    pass

vec = sorted(vec, key=lambda x: x[0])
vec = vec[1:11]
words = list(map(lambda x: x[1], vec))

print(words)
