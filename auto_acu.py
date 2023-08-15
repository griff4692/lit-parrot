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
from nltk import word_tokenize
from tqdm import tqdm
import backoff
from autoacu import A3CU, A2CU

from oa_secrets import OA_KEY, OA_ORGANIZATION

openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY

EXPERIMENTS = [
    # ['llama_incr', 's2l_llama', 's2l'],
    ['llama_straight', 's2l_llama_gpt4_selection', 'summarize'],
    ['llama_1', 'length_llama', '1'],
    ['llama_2', 'length_llama', '2'],
    ['llama_3', 'length_llama', '3'],
    ['llama_4', 'length_llama', '4'],
]


@backoff.on_exception(backoff.expo,
                      (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError),
                      max_tries=3)
def chatgpt(messages, model='gpt-4', max_tokens=368):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']


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

    # a3cu = A3CU(device=0)  # the GPU device to use
    a3cu = A2CU(device=1)

    # path = os.path.expanduser('~/seahorse/main_ideas/main_ideas_final/checkpoint-150')
    # pipe = pipeline("text-classification", model=path, device=args.device)

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

    acu_dir = os.path.expanduser(f'~/lit-parrot/out/eval/{args.experiment}/acu')
    os.makedirs(acu_dir, exist_ok=True)

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

    for exp in EXPERIMENTS:
        preds = [
            get_pred(exp, id) for id in shared_ids
        ]
        print(len(preds))
        recall_scores, prec_scores, f1_scores = a3cu.score(
            references=references,
            candidates=preds,
            batch_size=16,  # the batch size for ACU generation
            output_path=None  # the path to save the evaluation results
        )
        print(exp)
        print(f"Recall: {np.mean(recall_scores):.4f}, Precision {np.mean(prec_scores):.4f}, F1: {np.mean(f1_scores):.4f}")
        print('\n\n\n\n\n\n')
