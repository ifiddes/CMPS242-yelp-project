from toil.common import Toil
from toil.job import Job
from collections import Counter, defaultdict
import argparse
import itertools
import pandas as pd
import numpy as np
import simplejson as json
import enchant
from enchant.tokenize import get_tokenizer


def parse_args():
    parser = Job.Runner.getDefaultArgumentParser()
    parser.add_argument('review_json')
    parser.add_argument('output_json')
    return parser.parse_args()


def load_data(job, j):
    j = job.fileStore.readGlobalFile(j)
    with open(j) as f:
        return pd.DataFrame(json.loads(line) for line in f)


def parse_row(review_row, tokenizer):
    """returns spell checked words"""
    counts = Counter()
    for word, pos in tokenizer(review_row.text):
        counts[word] += 1
    return counts


def analyze_reviews(job, review_sub_df):
    d = enchant.Dict('en_US')
    tokenizer = get_tokenizer('en_US')
    c = []
    for _, review_row in review_sub_df.iterrows():
        c.append([review_row.review_id, parse_row(review_row, tokenizer)])
    return c


def setup(job, review_json):
    """construct the groups"""
    review_data = load_data(job, review_json)
    # start the splitting process
    results_holder = []
    for _, review_sub_df in review_data.groupby(np.arange(len(review_data)) // 250):
            results_holder.append(job.addChildJobFn(analyze_reviews, review_sub_df).rv())
    return job.addFollowOnJobFn(merge_vals, results_holder).rv()


def merge_vals(job, results_holder):
    c = {}
    for l in results_holder:
        for name, counts in l:
            c[name] = counts
    return c


if __name__ == '__main__':
    args = parse_args()
    with Toil(args) as toil:
        if not toil.options.restart:
            review_json = toil.importFile('file://' + args.review_json)
            job = Job.wrapJobFn(setup, review_json)
            results = toil.start(job)
        else:
            results = toil.restart()
        with open(args.output_json, 'w') as outf:
            for x, y in results.iteritems():
                z = {'parsed_counts': y, 'review_id': x}
                json.dumps(z, outf)
                outf.write('\n')

