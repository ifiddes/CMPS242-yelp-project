#!/usr/bin/python

import pandas as pd
import numpy as np
import json
from collections import Counter
import re
import math

def process_text(text_data):
    words = text_data.split()
    cnt = Counter(words)
    return cnt


print "Start loading data"
f = open('yelp_academic_dataset_review.json', 'r')

# This code is so that we can work with a small dataset first.
# Will need to be updated
ind = 1
total_reviews = 0
category_reviews = [0,0,0,0,0]
rows_dict = []
summary_words = []
for line in f:
    inter_dict = json.loads(line)
    rows_dict.append(inter_dict)
    if ind % 1000 ==0 and ind <= 2000000:
        training_data = pd.DataFrame(rows_dict)
        rows_dict = []

        grouped =  training_data.groupby('stars')
        idx = 0
        for name, group in grouped:
            category_reviews[idx] += len(group['text'])
            total_reviews += len(group['text'])
            idx += 1

        inter_words = []
        for name, group in grouped:
            word_cloud = group['text'].apply(process_text).sum()
            inter_words.append(word_cloud)
        if ind == 1000:
            for idx in range(len(inter_words)):
                summary_words.append(inter_words[idx])
        else:
            for idx in range(len(inter_words)):
                summary_words[idx] += inter_words[idx]
    ind += 1
    print ind
f.close()
print "End loading data"
test_data = pd.DataFrame(rows_dict)

prior = []
print "Compute priors from number of samples from each class"
for idx in range(len(category_reviews)):
    prior_val = float(category_reviews[idx])/total_reviews
    prior.append(prior_val)
    print prior_val

print "Remove all words that don't start with a character, and contain values other than characters or digits as well as some special characters. Print size of word cloud"
curated_summary = []
for ind in range(len(summary_words)):
    word_cloud = {}
    for word in summary_words[ind]:
        m = re.match(r'^[A-Za-z][A-Za-z\'!0-9]+$', word)
        if m:
           word_cloud[word] = summary_words[ind][word]
    print len(word_cloud)
    curated_summary.append(word_cloud)

# Remove stop words and create a stop words dict
stop_words = {}
f = open('stopwords_en.txt', 'r')
for line in f:
    w = line.rstrip('\r\n')
    stop_words[w] = 1
    for ind in range(len(curated_summary)):
        if w in curated_summary[ind]:
           del curated_summary[ind][w]
f.close()

print "Remove stop words. Print dictionary size"
for ind in range(len(curated_summary)):
    print len(curated_summary[ind])

print "For common feature set, generate union of words in all categories"
for ind1 in range(len(curated_summary)):
    for word in curated_summary[ind1]:
        for ind2 in range(len(curated_summary)):
            if ind1 == ind2:
               continue
            if not( word in curated_summary[ind2]):
               curated_summary[ind2][word] = 0

print "Number of words considered"
print len(curated_summary[1])

print "Compute conditionals. Use one-laplace smoothing"
# Perform one laplace smoothing
total_word_counts = []
for ind in range(len(curated_summary)):
    cnt = sum(curated_summary[ind].values())
    for word in curated_summary[ind]:
        curated_summary[ind][word] =  float(curated_summary[ind][word] + 1)/(len(curated_summary[ind]) + cnt)
print "Training phase complete"

print "Test phase"

def predict_text(grp):
    predictions = []
    for review in grp:
        classes = []
        for ind in range(len(prior)):
            classes.append(math.log(prior[ind]))
        words = review.split()
        for word in words:
            m = re.match(r'^[A-Za-z][A-Za-z\'!0-9]+$', word)
            if not m:
               continue
            if word in stop_words:
               continue
            if not(word in curated_summary[0]):
               continue
            for ind in range(len(curated_summary)):
                classes[ind] += math.log( curated_summary [ind][word] )

        max_class = 1
        max_val = classes[0]
        ind = 1
        while ind < len(curated_summary):
              if classes[ind] > max_val:
                 max_val = classes[ind]
                 max_class = ind + 1
              ind += 1

        predictions.append(max_class)

    return predictions

grouped = test_data.groupby('stars')
group_wise_predictions = grouped['text'].apply(predict_text)
ind = 1
correct = 0
for predictions in group_wise_predictions:
    for value in predictions:
        if value == ind:
           correct += 1
    ind += 1

print correct, len(test_data)
print "Accuracy: " + str(float(correct)/len(test_data))
print "Test phase complete"
