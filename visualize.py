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
    ['llama_chat_incr', 's2l_llama_chat', 's2l'],
    ['llama_chat_straight', 's2l_llama_chat', 'summarize'],
    ['llama_incr', 's2l_llama', 's2l'],
    ['llama_straight', 's2l_llama', 'summarize'],
    ['falcon_incr', 's2l_falcon', 's2l'],
    ['falcon_straight', 's2l_falcon', 'summarize'],
    ['llama_chat_1', 'length_llama_chat', '1'],
    ['llama_chat_2', 'length_llama_chat', '2'],
    ['llama_chat_3', 'length_llama_chat', '3'],
    ['llama_chat_4', 'length_llama_chat', '4'],
    ['llama_1', 'length_llama', '1'],
    ['llama_2', 'length_llama', '2'],
    ['llama_3', 'length_llama', '3'],
    ['llama_4', 'length_llama', '4'],
    ['falcon_1', 'length_falcon', '1'],
    ['falcon_2', 'length_falcon', '2'],
    ['falcon_3', 'length_falcon', '3'],
    ['falcon_4', 'length_falcon', '4'],
]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--experiment', default='default')
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    overwrite = args.overwrite

    rouge = load('rouge', keep_in_memory=True)

    eval_dir = os.path.expanduser(f'~/lit-parrot/out/eval/{args.experiment}/{args.dimension}')
    os.makedirs(eval_dir, exist_ok=True)

    print('Reading in dataset...')
    if args.dataset == 'cnn':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
    else:
        dataset = load_dataset(args.dataset, split='test')

    id2article = dict(zip(dataset['id'], dataset['article']))
    id2reference = dict(zip(dataset['id'], dataset['highlights']))

    def get_fns(info):
        suffix = info[-1]
        return list(glob('out/adapter_v2/' + info[1] + '/' + f'/*{suffix}.txt'))

    fns = [
        get_fns(info) for info in EXPERIMENTS
    ]
    ids_to_preds = defaultdict(dict)
    for fn in fns:
        id, task = fn.replace('.txt', '').split('/')[-1].split('_')
        with open(fn, 'r') as fd:
            pred = fd.read().strip()
            ids_to_preds[id][f'length_{task}'] = pred
    ids = list(ids_to_preds.keys())
    for id in ids:
        s2l_fn = os.path.join(s2l_dir, f'{id}_s2l.txt')
        if os.path.exists(s2l_fn):
            with open(s2l_fn, 'r') as fd:
                pred = fd.read().strip().split('\n')[-1]
                ids_to_preds[id]['s2l'] = pred
        else:
            print(f'Missing {s2l_fn}')
            ids_to_preds.pop(id)

    order = [
        's2l_llama',
        's2l_llama_chat',
        'straight_llama',
        'straight_llama_chat',
        'length_1_llama',
        'length_1_llama',
        'length_2_llama',
        'length_2_llama_chat',
        'length_3_llama',
        'length_3_llama_chat',
        'length_4_llama',
        'length_4_llama_chat',
    ]

    top_ranked = []

    avg_rank = {}
    tokens = {}
    rouges = {}
    for x in order:
        tokens[x] = []
        avg_rank[x] = []
        rouges[x] = []

    for id in ids_to_preds:
        out_fn = os.path.join(eval_dir, f'{id}.json')
        reference = id2reference[id]

        predictions = [
            ids_to_preds[id][task] for task in order
        ]

        for pred, type in zip(predictions, order):
            r1 = rouge.compute(predictions=[pred], references=[reference], rouge_types=['rouge1'])['rouge1']
            rouges[type].append(r1)

        print('S2L:', predictions[0])
        print('Length 1:', predictions[1])
        print('Length 2:', predictions[2])
        print('Length 3:', predictions[3])
        print('Length 4:', predictions[4])

        token_cts = [
            len(word_tokenize(x)) for x in predictions
        ]

        for task, ct in zip(order, token_cts):
            tokens[task].append(ct)

        article = id2article[id].strip()

    print('Tokens...')
    for task, cts in tokens.items():
        print(task + '\t' + str(np.mean(cts)))
