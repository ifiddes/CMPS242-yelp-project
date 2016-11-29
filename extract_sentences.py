import simplejson as json
import pandas as pd
import re
import unicodedata
import sys
import string
from orderedset import OrderedSet
# load data


with open('yelp_academic_dataset_review.json', 'r') as f:
    review_data = pd.DataFrame((json.loads(line) for line in f))


# load stop words
stop_words = {x.rstrip() for x in open('stopwords_en.txt')}

# regex to construct sentences
# http://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
regex = re.compile(u'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?)\s')

tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))


def parse_review(row, sentences, labels):
    for s in regex.split(row.text):
        # some reviews are str, some are unicode
        s = unicode(s)
        # split into words
        s = s.split()
        # lower case, remove punctuation
        s = OrderedSet(x.translate(tbl).lower() for x in s if len(x) > 0)
        s -= stop_words
        if len(s) < 5:
            continue
        # make string
        s = u' '.join(s) + u'\n'
        sentences.append(s)
        labels.append(row.stars)


train_sentences = []
train_labels = []
test_sentences = []
test_labels = []
for i, row in review_data.iterrows():
    if i <= 2000000:
        parse_review(row, train_sentences, train_labels)
    else:
        parse_review(row, test_sentences, test_labels)


# convert to sentence file
with open('train_sentences.txt', 'w') as outf:
    for s in train_sentences:
        outf.write(s.encode('utf8'))


with open('test_sentences.txt', 'w') as outf:
    for s in test_sentences:
        outf.write(s.encode('utf8'))


with open('train_labels.txt', 'w') as outf:
    for l in train_labels:
        outf.write(str(l) + '\n')


with open('test_labels.txt', 'w') as outf:
    for l in test_labels:
        outf.write(str(l) + '\n')

