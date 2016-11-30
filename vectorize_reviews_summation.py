#!/usr/bin/python

import sys
import codecs
import numpy as np

ip_fname = sys.argv[1]
vec_fname = sys.argv[2]
op_fname = sys.argv[3]
size = sys.argv[4]

embed_dict = {}
f = open(vec_fname, "r")
for line in f:
    words = line.split()
    word = words[0]
    lst = map(float, words[1:])
    embed_dict[word] = lst
f.close()

f1 = open(ip_fname, "r")
f2 = open(op_fname, "w")
idx = 1
count_words = 0
for line in f1:
    print idx
    words = line.split()
    review_vector = np.zeros((int(size)))
    for word in words:
        if word in embed_dict:
           review_vector += embed_dict[word]
           count_words += 1
    ind = 0
    review_vector /= count_words
    while ind < len(review_vector):
          f2.write(" ")
          f2.write(str(review_vector[ind]))
          ind += 1
    f2.write("\n")
    idx += 1

f1.close()
f2.close()
