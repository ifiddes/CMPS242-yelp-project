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


def parse_review(row, sentences):
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


train_sentences = []
for i, row in review_data[:2000000].iterrows():
        parse_review(row, train_sentences)


# convert to sentence file
with open('train_sentences.txt', 'w') as outf:
    for s in train_sentences:
        outf.write(s.encode('utf8'))



import sys
# the imports are broken
sys.path.insert(0, '/home/ifiddes/anaconda3/lib/python3.5/site-packages/word2vec')
from w2v import word2vec
embedder, dictionary = word2vec(files=['train_sentences.txt'],
                                num_processes=32,
                                save_dir='save_tmp')


def find_embeddings(sentences):
    embeddings = []
    for s in sentences:
        tokens = s.split()
        token_ids = dictionary.unigram_dictionary.get_ids(tokens)
        vectors = embedder.embed(token_ids)
        embeddings.append(np.sum(vectors, axis=0))
    return np.sum(embeddings, axis=0)


# for each training review, find the embeddings of each word. Sum these up.
# this produces a 1x500 vector for each training review
train_vectors = []
train_labels = []
for i, row in review_data[:2000000].iterrows():
    review_sentences = []
    parse_review(row, review_sentences)
    if len(review_sentences) == 0:
        continue
    train_vectors.append(find_embeddings(review_sentences))
    train_labels.append(row.stars)


test_vectors = []
test_labels = []
for i, row in review_data[2000000:].iterrows():
    review_sentences = []
    parse_review(row, review_sentences)
    if len(review_sentences) == 0:
        continue
    test_vectors.append(find_embeddings(review_sentences))
    test_labels.append(row.stars)


train_matrix = np.matrix(train_vectors)


# write to disk, adding the labels as the 1st column
with open('training_matrix.tsv', 'w') as outf:
    for l, r in zip(*[train_labels, train_vectors]):
        outf.write(str(l) + '\t')
        outf.write('\t'.join(map(str, r)) + '\n')


# write to disk, adding the labels as the 1st column
with open('test_matrix.tsv', 'w') as outf:
    for l, r in zip(*[test_labels, test_vectors]):
        outf.write(str(l) + '\t')
        outf.write('\t'.join(map(str, r)) + '\n')


# perform ordinal regression using the mord package
from mord import *
c = LogisticIT()
c.fit(train_matrix, train_labels)

results = []
for v, l in zip(*[test_vectors, test_labels]):
    pred_l = c.predict(v)
    results.append(pred_l)


from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

accuracy = 1.0 * len([x for x, y in zip(*[results, test_labels]) if x == y]) / len(results)

cm = confusion_matrix(test_labels, results)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix after using mord on the full data set\naccuracy = {:.2f}'.format(accuracy))
plt.colorbar()
tick_marks = np.arange(5)
plt.xticks(tick_marks, range(1, 6), rotation=45)
plt.yticks(tick_marks, range(1, 6))
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")


plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('mord_full_data.pdf')


# bring in ordinal logistic regression implementation made by Jacob
from ologr import *
logr = OrdinalLogisticRegressionIT()


# perform analysis on subset, train with 100k test with 10k
test_subset = test_vectors[:10000]
test_subset_labels = test_labels[:10000]
train_subset = train_matrix[:100000]
train_subset_labels = train_labels[:100000]

w, theta = logr.train(train_subset, train_subset_labels)

jacob_results = []
for v, l in zip(*[test_subset, test_subset_labels]):
    pred_l = logr.predict(w, theta,v)
    jacob_results.append(pred_l)

accuracy = 1.0 * len([x for x, y in zip(*[jacob_results, test_subset_labels]) if x == y]) / len(jacob_results)

cm = confusion_matrix(test_subset_labels, jacob_results)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix after using Jacob's implementation on a subset\naccuracy = {:.2f}".format(accuracy))
plt.colorbar()
tick_marks = np.arange(5)
plt.xticks(tick_marks, range(1, 6), rotation=45)
plt.yticks(tick_marks, range(1, 6))
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")


plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('jacob_subset.pdf')
