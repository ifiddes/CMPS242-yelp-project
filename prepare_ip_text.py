#!/usr/bin/python

import pandas as pd
import numpy as np
import json
from collections import Counter
import re
import codecs
import sys

training_text_fname = sys.argv[1]
training_labels_fname = sys.argv[2]
test_text_fname = sys.argv[3]
test_labels_fname = sys.argv[4]
number_training = sys.argv[5]
number_test = sys.argv[6]

print "Start loading data"
with open('yelp_academic_dataset_review.json', 'r') as f:
     review_data = pd.DataFrame(json.loads(line) for line in f)

training_data = review_data.ix[:2000000]
test_data = review_data.ix[2000001:]
print "End loading data"

f1 = codecs.open(training_text_fname, 'w', encoding='utf-8')
f2 = codecs.open(training_labels_fname, 'w', encoding='utf-8')
ind = 0
stars = 1
grouped = training_data.groupby('stars')
for name, group in grouped:
    for review in group['text']:
        review = review.replace('\n', ' ')
        f1.write(review)
        f1.write('\n')
        f2.write(str(stars))
        f2.write('\n')
        ind +=1
        if ind == int(number_training):
           ind = 0
           break
    stars += 1

f1.close()
f2.close()

f1 = codecs.open(test_text_fname, 'w', encoding='utf-8')
f2 = codecs.open(test_labels_fname, 'w', encoding='utf-8')
ind = 0
stars = 1
grouped = test_data.groupby('stars')
for name, group in grouped:
    for review in group['text']:
        review = review.replace('\n', ' ')
        f1.write(review)
        f1.write('\n')
        f2.write(str(stars))
        f2.write('\n')
        ind +=1
        if ind == int(number_test):
           ind = 0
           break
    stars += 1

f1.close()
f2.close()
