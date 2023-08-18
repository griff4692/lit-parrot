from collections import Counter
import os
from glob import glob
import argparse
import regex as re
import math

from collections import defaultdict
from datasets import load_dataset, load_from_disk
import numpy as np
import json
import pandas as pd

EXPERIMENTS = [
    # ['llama_incr', 's2l_llama', 's2l'],
    ['llama_straight', 's2l_llama', 'summarize'],
    ['llama_1', 'length_llama', '1'],
    ['llama_2', 'length_llama', '2'],
    ['llama_3', 'length_llama', '3'],
    ['llama_4', 'length_llama', '4'],
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--model_class', default='gpt4', choices=['llama', 'gpt4'])
    parser.add_argument('--max_examples', default=25, type=int)

    args = parser.parse_args()

    def get_pred(info, id):
        suffix = info[-1]
        if 's2l' in info[1]:
            fn = f'~/Desktop/s2l_results_8_14/{args.dataset}/{id}_{suffix}.txt'
        else:
            fn = f'~/Desktop/length_results_8_14/{args.dataset}/{id}_{suffix}.txt'
        fn = os.path.expanduser(fn)
        with open(fn, 'r') as fd:
            pred_lines = fd.readlines()
        pred_lines = [
            x.strip() for x in pred_lines if len(x.strip()) > 0
        ]
        return pred_lines[-1]


    def get_fns(info):
        suffix = info[-1]

        if 's2l' in info[1]:
            pattern = f'~/Desktop/s2l_results_8_14/{args.dataset}/*{suffix}.txt'
        else:
            pattern = f'~/Desktop/length_results_8_14/{args.dataset}/*{suffix}.txt'
        pattern = os.path.expanduser(pattern)

        fns = list(glob(pattern))
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
        np.random.seed(1444)
        np.random.shuffle(shared_ids)
        shared_ids = shared_ids[:args.max_examples]

    print('Reading in dataset...')
    if args.dataset == 'cnn':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
    elif args.dataset == 'xsum':
        dataset = load_dataset(args.dataset, split='test')
    else:
        dataset = load_from_disk(os.path.expanduser('~/nyt_edu_alignments'))['test']
        dataset = dataset.rename_columns({
            'article_untok': 'document',
            'abstract_untok': 'summary'
        })

    id2article = dict(zip(dataset['id'], dataset['article'] if 'article' in dataset.features else dataset['document']))
    id2reference = dict(
        zip(dataset['id'], dataset['highlights'] if 'highlights' in dataset.features else dataset['summary']))

    references = [
        id2reference[id] for id in shared_ids
    ]

    outputs = []
    meta = []
    for id in shared_ids:
        row = f'ID: {id}\n\n'
        s2l = get_pred(EXPERIMENTS[0], id)
        compare_idx = int(np.random.randint(1, len(EXPERIMENTS)))
        compare_exp = EXPERIMENTS[compare_idx]
        compare_pred = get_pred(compare_exp, id)

        article = id2article[id]
        s2l_first = np.random.random() > 0.5
        row += f'Article:\n{article}\n\n'
        if s2l_first:
            row += f'Summary A: {s2l}\n\nSummary B: {compare_pred}\n\n'
        else:
            row += f'Summary A: {compare_pred}\n\nSummary B: {s2l}\n\n'

        row += 'Alex Preference:\n\nGriffin Preference:\n\n'
        outputs.append(row)
        meta.append({
            'comparison': compare_exp[0],
            's2l_is_summary_a': s2l_first,
        })
        
    meta = pd.DataFrame(meta)
    delim = '*' * 100 + '\n\n'
    with open('human_eval.txt', 'w') as fd:
        fd.write(delim.join(outputs))
    
    meta.to_csv('human_meta.csv', index=False)
