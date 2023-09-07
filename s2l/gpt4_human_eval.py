import argparse

import pandas as pd
from datasets import load_dataset
import numpy as np
np.random.seed(1992)

from data_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--max_examples', default=100, type=int)

    args = parser.parse_args()

    info = ['human_straight_dense_v2']
    experiment_fns = get_gpt4_fns(info)

    shared_ids = set([x[0] for x in experiment_fns])

    print('Reading in dataset...')
    dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')

    id2article = dict(zip(dataset['id'], dataset['article'] if 'article' in dataset.features else dataset['document']))
    id2reference = dict(
        zip(dataset['id'], dataset['highlights'] if 'highlights' in dataset.features else dataset['summary']))

    references = [
        id2reference[id] for id in shared_ids
    ]

    outputs = []
    metas = []
    for j, id in enumerate(shared_ids):
        dense = get_gpt4_preds(info, id, return_missing=False)[0].strip()
        dense_first = np.random.random() < 0.5

        baseline = get_gpt4_preds(['baseline_human_test'], id, return_missing=False)
        assert len(baseline) == 1
        baseline = baseline[0].strip()

        row = f'ID: {id}\n\n'
        article = id2article[id]
        row += f'Article:\n{article}\n\n'
        if dense_first:
            meta = {'idx': j, 'id': id, 'Summary 1': 'dense', 'Summary 2': 'baseline'}
            row += f'Summary 1: {dense}\n\nSummary 2: {baseline}\n\n'
        else:
            meta = {'idx': j, 'id': id, 'Summary 1': 'baseline', 'Summary 2': 'dense'}
            row += f'Summary 1: {baseline}\n\nSummary 2: {dense}\n\n'
        metas.append(meta)

        row += 'Preference:\n'
        row += 'Reason:\n\n'
        outputs.append(row)

    metas = pd.DataFrame(metas)
    metas.to_csv('gpt4_human_9_5.csv', index=False)

    delim = '*' * 75 + '\n\n'
    with open('gpt4_human_9_5.txt', 'w') as fd:
        fd.write(delim.join(outputs))
