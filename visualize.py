from collections import defaultdict
import numpy as np
import os
from evaluate import load
from time import sleep
from collections import Counter
from glob import glob

from datasets import load_dataset
from nltk import word_tokenize
import json


EXPERIMENTS = [
    ['llama_incr', 's2l_llama', 's2l'],
    ['llama_straight', 's2l_llama', 'summarize'],
    ['llama_1', 'length_llama', '1'],
    ['llama_2', 'length_llama', '2'],
    ['llama_3', 'length_llama', '3'],
    ['llama_4', 'length_llama', '4'],
]


def get_pred(info, id):
    suffix = info[-1]
    fn = 'out/adapter_v2/' + info[1] + f'/results/{id}_{suffix}.txt'
    with open(fn, 'r') as fd:
        pred_lines = fd.readlines()
    pred_lines = [
        x.strip() for x in pred_lines if len(x.strip()) > 0
    ]
    return pred_lines[-1]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--max_examples', default=10, type=int)

    args = parser.parse_args()

    print('Reading in dataset...')
    if args.dataset == 'cnn':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
    else:
        dataset = load_dataset(args.dataset, split='test')

    id2article = dict(zip(dataset['id'], dataset['article']))
    id2reference = dict(zip(dataset['id'], dataset['highlights']))

    def get_fns(info):
        suffix = info[-1]
        fns = list(glob('out/adapter_v2/' + info[1] + f'/results/*{suffix}.txt'))
        ids = [
            fn.split('/')[-1].replace('.txt', '').split('_')[0] for fn in fns
        ]
        return list(zip(ids, fns))

    experiment_fns = [
        get_fns(info) for info in EXPERIMENTS
    ]

    print([
        (EXPERIMENTS[i][0], len(experiment_fns[i])) for i in range(len(experiment_fns))
    ])

    shared_ids = set([x[0] for x in experiment_fns[0]])
    for i in range(2, len(experiment_fns)):
        shared_ids.intersection(set([x[0] for x in experiment_fns[i]]))

    shared_ids = list(sorted(list(shared_ids)))
    assert len(shared_ids) >= args.max_examples

    if len(shared_ids) > args.max_examples:
        np.random.seed(1992)
        np.random.shuffle(shared_ids)
        shared_ids = shared_ids[:args.max_examples]

    vis = []
    for id in shared_ids:
        reference = id2reference[id]

        predictions = [
            info[0] + ': ' + get_pred(info, id) for info in EXPERIMENTS
        ]

        vis.append('\n'.join(predictions))

    vis = ('\n\n' + '*' * 50 + '\n\n').join(vis)

    print(vis)