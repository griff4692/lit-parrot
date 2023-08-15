from collections import Counter
import os
from glob import glob
import argparse
import regex as re
import math

from datasets import load_dataset
import openai
import numpy as np
import json
from nltk import word_tokenize
from tqdm import tqdm
import backoff

from oa_secrets import OA_KEY, OA_ORGANIZATION

openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY

EXPERIMENTS = [
    ['llama_straight', 's2l_llama', 'summarize'],
    ['llama_1', 'length_llama', '1'],
    ['llama_2', 'length_llama', '2'],
    ['llama_3', 'length_llama', '3'],
    ['llama_4', 'length_llama', '4'],
]


@backoff.on_exception(backoff.expo,
                      (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError),
                      max_tries=25)
def chatgpt(messages, model='gpt-4', max_tokens=128):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    PROMPT = """
    You are shown 5 summaries above.  Please label each summary based on its conciseness and informativeness. Select one of the following categories for each.
    
    - Too short: Lacks critical information.
    - Just right: Includes the right amount of critical information without being overly verbose.
    - Too long: the summary includes extraneous, unnecessary details or is overly verbose.
    
    Do not explain. Return a list of JSON strings representing the labels: "too short", "just right", "too long".
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--experiment', default='default')
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--max_examples', default=20, type=int)

    args = parser.parse_args()

    overwrite = args.overwrite


    def get_pred(info, id):
        suffix = info[-1]
        fn = 'out/adapter_v2/' + info[1] + f'/results/{id}_{suffix}.txt'
        with open(fn, 'r') as fd:
            pred_lines = fd.readlines()
        pred_lines = [
            x.strip() for x in pred_lines if len(x.strip()) > 0
        ]
        return pred_lines[-1]


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

    acu_dir = os.path.expanduser(f'~/lit-parrot/out/eval/{args.experiment}/scale')
    os.makedirs(acu_dir, exist_ok=True)

    print('Reading in dataset...')
    if args.dataset == 'cnn':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
    else:
        dataset = load_dataset(args.dataset, split='test')

    id2article = dict(zip(dataset['id'], dataset['article']))
    id2reference = dict(zip(dataset['id'], dataset['highlights']))

    from collections import defaultdict

    exp2scales = defaultdict(list)
    exp2toks = defaultdict(list)

    for id in tqdm(shared_ids):
        out_fn = os.path.join(acu_dir, f'{id}.json')
        article = id2article[id]

        predictions = [
            get_pred(info, id) for info in EXPERIMENTS
        ]

        token_cts = [
            len(word_tokenize(x)) for x in predictions
        ]

        for exp, tok in zip(EXPERIMENTS, token_cts):
            exp2toks[exp[0]].append(tok)

        pred_strs = '\n\n'.join([
            f'Summary {i + 1}: {x}' for i, x in enumerate(predictions)
        ])

        token_cts = [
            len(word_tokenize(x)) for x in predictions
        ]

        if os.path.exists(out_fn) and not args.overwrite:
            print(f'Loading in Scales from {out_fn}')
            with open(out_fn, 'r') as fd:
                scales = json.load(fd)
                scales = [x.lower() for x in scales]
        else:
            messages = [
                # Boost its ego first
                {'role': 'system',
                 'content': 'You value concise and non-redundant summaries regardless of their length.'},
                {'role': 'user', 'content': f'{pred_strs}\n\n{PROMPT}'}  # Article: {article}\n\n
            ]

            scales = json.loads(chatgpt(messages))
            scales = [list(x.values())[0].lower() if type(x) == dict else x.lower() for x in scales]

            with open(out_fn, 'w') as fd:
                json.dump(scales, fd)

        for exp, scale in zip(EXPERIMENTS, scales):
            exp2scales[exp[0]].append(scale)

        ordered_by_len = sorted(exp2toks, key=lambda x: np.mean(exp2toks[x]))
        for k in ordered_by_len:
            print(k, Counter(exp2scales[k]).most_common(), np.mean(exp2toks[k]))
