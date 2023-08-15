from collections import Counter
import os
from glob import glob
import argparse
import regex as re
import math

from collections import defaultdict
from datasets import load_dataset, load_from_disk
import openai
from evaluate import load
import numpy as np
import json
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
import backoff
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
from scipy.stats import pearsonr
from nltk.util import ngrams
from rouge_utils import *
from fragments import compute_frags


import spacy as nlp


EXPERIMENTS = [
    # ['llama_incr', 's2l_llama', 's2l'],
    ['llama_straight', 's2l_llama_gpt4_selection', 'summarize'],
    ['llama_1', 'length_llama', '1'],
    ['llama_2', 'length_llama', '2'],
    ['llama_3', 'length_llama', '3'],
    ['llama_4', 'length_llama', '4'],
]


def num_ents(text, nlp):
    return len(list(set(list(nlp(text).ents))))


def redundancy(text):
    toks = word_tokenize(text)
    unigrams = list(ngrams(toks, 1))
    bigrams = list(ngrams(toks, 2))
    trigrams = list(ngrams(toks, 3))

    return {
        'unique_unigrams': len(set(unigrams)) / len(unigrams),
        'unique_bigrams': len(set(bigrams)) / len(bigrams),
        'unique_trigrams': len(set(trigrams)) / len(trigrams),
    }


def compute_exp(nlp, name, sources, source_tokens, preds):
    exp_stats = defaultdict(list)
    tokens = [
        len(word_tokenize(pred)) for pred in preds
    ]
    exp_stats['tokens'] = tokens
    exp_stats['length_correlation'] = pearsonr(tokens, source_tokens)[0]

    # Create the histogram
    sns.histplot(tokens, bins=20, kde=True)

    plt.title(f'Distribution of tokens for {name}')

    # Save the plot to 'save.png'
    plt.savefig(f"{name}_tokens_hist.png")

    plt.clf()

    frags = [compute_frags({'source': source, 'prediction': pred}) for source, pred in zip(sources, preds)]

    exp_stats['coverage'] = [f['coverage'] for f in frags]
    exp_stats['density'] = [f['density'] for f in frags]

    out = [redundancy(pred) for pred in preds]
    exp_stats['num_ents'] = [num_ents(pred, nlp) for pred in preds]
    exp_stats['num_ents_per_token'] = [a / b for a, b in zip(exp_stats['num_ents'], exp_stats['tokens'])]
    exp_stats['unique_unigrams'] = [x['unique_unigrams'] for x in out]
    exp_stats['unique_bigrams'] = [x['unique_bigrams'] for x in out]
    exp_stats['unique_trigrams'] = [x['unique_trigrams'] for x in out]

    all_pred_sents = [sent_tokenize(pred) for pred in preds]
    for source, pred_sents in zip(sources, all_pred_sents):
        source_sents = sent_tokenize(source)
        source_sents_no_stop = [
            remove_stopwords(x) for x in source_sents
        ]
        aligned_idxs = []
        for sent in pred_sents:
            source_idxs, _ = gain_rouge(sent, source_sents_no_stop, max_steps=5)
            aligned_idxs.append(source_idxs)
        fusion_score = np.mean([len(x) for x in aligned_idxs])
        avg_rank = float(np.mean(list(chain(*aligned_idxs))))

        exp_stats['fusion'].append(fusion_score)
        exp_stats['avg_sent_rank'].append(avg_rank)
    print(name)
    for k, v in exp_stats.items():
        print(k, np.mean(v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--experiment', default='default')
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--max_examples', default=100, type=int)
    parser.add_argument('--device', default=1, type=int)

    args = parser.parse_args()

    args.experiment += '_' + args.dataset

    overwrite = args.overwrite

    rouge = load('rouge', keep_in_memory=True)

    nlp = spacy.load("en_core_web_sm")

    def get_pred(info, id):
        suffix = info[-1]
        fn = 'out/adapter_v2/' + info[1] + f'/results/{args.dataset}/{id}_{suffix}.txt'
        with open(fn, 'r') as fd:
            pred_lines = fd.readlines()
        pred_lines = [
            x.strip() for x in pred_lines if len(x.strip()) > 0
        ]
        return pred_lines[-1]

    def get_fns(info):
        suffix = info[-1]
        fns = list(glob('out/adapter_v2/' + info[1] + f'/results/{args.dataset}/*{suffix}.txt'))
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

    stats_dir = os.path.expanduser(f'~/lit-parrot/out/eval/stats')
    os.makedirs(stats_dir, exist_ok=True)

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

    sources = [id2article[id] for id in shared_ids]
    source_tokens = [len(word_tokenize(source)) for source in sources]
    for exp in EXPERIMENTS:
        preds = [get_pred(exp, id) for id in shared_ids]
        compute_exp(nlp, exp[0], sources, source_tokens, preds)

    compute_exp(nlp, 'reference', sources, source_tokens, references)
