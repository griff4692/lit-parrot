import argparse

import pandas as pd
from datasets import load_dataset
import numpy as np
np.random.seed(1992)

from data_utils import *

NAMES = ['Initial', 'Step 1', 'Step 2', 'Step 3', 'Step 4']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--max_examples', default=100, type=int)

    args = parser.parse_args()

    info = ['human_dense_v3']
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
        dense = get_gpt4_preds(info, id, return_missing=False)[:5]
        order = np.arange(5)
        np.random.shuffle(order)

        ordered_names = [NAMES[rand_idx] for rand_idx in order]

        dense_random = [dense[rand_idx] for rand_idx in order]
        meta = {'idx': j, 'id': id, 'order': ','.join([str(int(x)) for x in order])}
        for rank, name in enumerate(ordered_names):
            meta[f'Summary {rank + 1}'] = name
        metas.append(meta)
        row = f'ID: {id}\n\n'
        article = id2article[id]
        row += f'Article:\n{article}\n\n'
        for i, d in enumerate(dense_random):
            row += f'Summary {i + 1}: {d}\n\n'

        row += 'Preference:\n'
        row += 'Reason:\n\n'
        outputs.append(row)

    metas = pd.DataFrame(metas)
    metas.to_csv('dense_human_8_25.csv', index=False)

    delim = '*' * 75 + '\n\n'
    with open('dense_human_8_25.txt', 'w') as fd:
        fd.write(delim.join(outputs))
