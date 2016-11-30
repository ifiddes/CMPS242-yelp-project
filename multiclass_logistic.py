#!/usr/bin/python
import os
import math
import re
import numpy as np
import collections as ct
from scipy.special import expit
import scipy.optimize as optimizer
import sys

train_fname = sys.argv[1]
test_fname = sys.argv[2]
train_category_num = int(sys.argv[3])
test_category_num = int(sys.argv[4])

print "Construct matrices for optimisation"
print "Construct X matrix"
ind = 0
f1 = open(train_fname, 'r')
for line in f1:
    comp = map(float, line.split())
    const = np.array([1])
    res = np.concatenate((const, np.asarray(comp))).T
    if ind == 0:
       x = res
    else:
       x = np.column_stack((x, res))
    ind += 1
f1.close()
print(x.shape)

print"Construct w matrix"
w_t = np.ones( (5,201) )
#w.fill(0.5)

print("Construct t matrix")
category = np.identity(5)
ind = 1
t = category[0]
class_num = 1
while class_num <= 5:
    t = np.row_stack((t, category[class_num-1]))
    ind += 1
    if ind == train_category_num:
       ind = 0
       class_num += 1
print(t.shape)

def cost_func(w_t, x, category, train_category_num, size):
    cost = 0
    for class_num in range(5):
          start = class_num*train_category_num
          end = (class_num+1)*train_category_num

          #print (w_t.shape)
          #print (x[:, start].shape)
          #print(type(x[0,0]))

          srt_idx = class_num*size
          end_idx = (class_num+1)*size
          #print srt_idx, end_idx
          cost += np.sum(w_t[srt_idx:end_idx].dot(x[:, start:end]))

    for class_num in range(5):
        start = class_num*size
        end = (class_num+1)*size

        if class_num ==0:
           normalized = w_t[start:end].dot(x)
        else:
            normalized = np.row_stack((normalized,w_t[start:end].dot(x)))

    #print np.sum(normalized, axis = 0).shape
    #print(normalized.shape)

    max_vals = np.amax(normalized, axis=0)
    sum_max = np.sum(max_vals)

    normalized_reduced = np.subtract(normalized, max_vals)
    total_val = np.sum(np.log(np.sum(np.exp(normalized_reduced), axis = 0)))
    total_val += sum_max
    cost -= total_val
    #print "Cost"
    #print(val)
    print(-cost)
    return(-cost)

print "Optimize"
res = optimizer.minimize(cost_func, w_t, method='BFGS',args=(x, category, train_category_num, 201), options={'xtol': 1e-8, 'disp': True} )
w_optimized = res.x
print(w_optimized)
print(len(w_optimized))

f1 = open(test_fname, 'r')
class_num = 0
ind = 0
size = 201
correct = 0
classify = 0
count = 0
for line in f1:
    comp = map(float, line.split())
    const = np.array([1])
    comp = np.concatenate((const, np.asarray(comp))).T
    max_val = 0
    for idx in range(5):
        start = idx*size
        end = (idx+1)*size
        res = w_optimized[start:end].dot(comp)
        print res, max_val
        if res > max_val:
           max_val = res
           classify = idx
    print classify, class_num
    if classify == class_num:
       correct += 1
    if ind == test_category_num:
       ind = 0
       class_num += 1
    ind += 1
    count += 1
f1.close()
print(correct)
print(count)
