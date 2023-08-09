from collections import Counter
import os
from glob import glob
import argparse
import regex as re
import math

from collections import defaultdict
from datasets import load_dataset, load_from_disk
import openai
from transformers import pipeline
from evaluate import load
import numpy as np
import json
from nltk import word_tokenize
from tqdm import tqdm
import backoff

from oa_secrets import OA_KEY, OA_ORGANIZATION


openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY

EXPERIMENTS = [
    # ['llama_incr', 's2l_llama', 's2l'],
    ['llama_straight', 's2l_llama', 'summarize'],
    ['llama_1', 'length_llama', '1'],
    ['llama_2', 'length_llama', '2'],
    ['llama_3', 'length_llama', '3'],
    ['llama_4', 'length_llama', '4'],
]


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError), max_tries=3)
def chatgpt(messages, model='gpt-4', max_tokens=368):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    PROMPT = """
    You are tasked with create a plan for summarizing the below Article. Specifically, you need to create a checklist of Atomic Fact Units (ACUs) which should be included in a potential summary.

    Strictly adhere to the following definitions and rules:
    - An ACU is a standalone fact from the Article which does not contain pronouns or any ambiguous references.
    - Return an ordered list of at most 15 salient ACUs.
    - Stop once all critically important ACUs are listed.
    - Order by importance, NOT by position in the Article.
    - An ACU should be no more than 8 words long.
    - There should be no repeated or overlapping content across ACUs.

    Generate a list from most to least critical.
    """

    MATCH_PROMPT = """
    You are tasked with determining the presence or absence of a set of Atomic Content Units (ACUs) in a provided Summary.
    
    List out the ACUs which are covered by the Summary.
    - Do not explain your answer.
    - Respond with the number of the specific matching ACUs.
    - Return a JSON list of integers.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--experiment', default='default')
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--max_examples', default=20, type=int)
    parser.add_argument('--device', default=1, type=int)

    args = parser.parse_args()

    args.experiment += '_' + args.dataset

    overwrite = args.overwrite

    rouge = load('rouge', keep_in_memory=True)

    # path = os.path.expanduser('~/seahorse/main_ideas/main_ideas_final/checkpoint-150')
    # pipe = pipeline("text-classification", model=path, device=args.device)

    def get_pred(info, id):
        suffix = info[-1]
        if args.dataset == 'cnn':
            fn = 'out/adapter_v2/' + info[1] + f'/results/{id}_{suffix}.txt'
        else:
            fn = 'out/adapter_v2/' + info[1] + f'/results/{args.dataset}/{id}_{suffix}.txt'
        with open(fn, 'r') as fd:
            pred_lines = fd.readlines()
        pred_lines = [
            x.strip() for x in pred_lines if len(x.strip()) > 0
        ]
        return pred_lines[-1]

    def get_fns(info):
        suffix = info[-1]
        if args.dataset == 'cnn':
            fns = list(glob('out/adapter_v2/' + info[1] + f'/results/*{suffix}.txt'))
        else:
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

    id2article = dict(zip(dataset['id'], dataset['article'] if 'article' in dataset else dataset['document']))
    id2reference = dict(zip(dataset['id'], dataset['highlights'] if 'highlights' in dataset else dataset['summary']))

    exp2dcg = defaultdict(list)
    exp2toks = defaultdict(list)
    rouges = defaultdict(list)
    horses = defaultdict(list)

    for id in tqdm(shared_ids):
        out_fn = os.path.join(acu_dir, f'{id}.json')
        match_fn = os.path.join(acu_dir, f'{id}_matches.json')
        article = id2article[id]

        predictions = [
            get_pred(info, id) for info in EXPERIMENTS
        ]

        # seahorse_inputs = [
        #     {'text': p, 'text_pair': article} for p in predictions
        # ]

        # seahorse_preds = pipe(seahorse_inputs, max_length=2048, truncation=True, batch_size=32)
        # seahorse_probs = []
        # for exp, score in zip(EXPERIMENTS, seahorse_preds):
        #     if score['label'] == 0:
        #         pos_prob = 1 - score['score']
        #     else:
        #         pos_prob = score['score']
        #     horses[exp[0]].append(pos_prob)

        for pred, exp in zip(predictions, EXPERIMENTS):
            r1 = rouge.compute(
                predictions=[pred], references=[id2reference[id]], rouge_types=['rouge1']
            )['rouge1']
            rouges[exp[0]].append(r1)

        token_cts = [
            len(word_tokenize(x)) for x in predictions
        ]

        for exp, tok in zip(EXPERIMENTS, token_cts):
            exp2toks[exp[0]].append(tok)

        print(out_fn, os.path.exists(out_fn))
        if os.path.exists(out_fn) and not args.overwrite:
            print(f'Loading in ACUs from {out_fn}')
            with open(out_fn, 'r') as fd:
                acus = json.load(fd)

            if type(acus) == str:
                acus = acus.split('\n')
                acus = [
                    re.sub(r'^\d+\.\s+', '', acu) for acu in acus
                ]
            with open(match_fn, 'r') as fd:
                matches = json.load(fd)
        else:
            messages = [
                # Boost its ego first
                {'role': 'system', 'content': 'You are a concise and non-redundant extractor of important content for summarization.'},
                {'role': 'user', 'content': f'{PROMPT}\n\nArticle: {article}'}
            ]

            try:
                output = chatgpt(messages)
            except:
                print('Skipping this for now.')
                continue
            acus = output.split('\n')
            acus = [
                re.sub(r'^\d+\.\s+', '', acu) for acu in acus
            ]
            with open(out_fn, 'w') as fd:
                json.dump(acus, fd)
        if os.path.exists(match_fn) and not args.overwrite:
            with open(match_fn, 'r') as fd:
                matches = json.load(fd)
        else:
            matches = []
            acu_str = '\n'.join([
                f'{i + 1}. {acu}' for i, acu in enumerate(acus)
            ])

            valid = True
            for pred in predictions:
                messages = [
                    # Boost its ego first
                    {'role': 'system',
                     'content': 'You are a concise and non-redundant extractor of important content for summarization.'},
                    {'role': 'user', 'content': f'{MATCH_PROMPT}\n\nACUs:\n{acu_str}\n\nSummary: {pred}'}
                ]

                try:
                    output = chatgpt(messages, max_tokens=64)
                except:
                    print('Skipping this for now.')
                    valid = False
                    break

                matches.append(json.loads(output))

            if valid:
                with open(match_fn, 'w') as fd:
                    json.dump(matches, fd)

        for exp, match in zip(EXPERIMENTS, matches):
            # dcg = sum([1 / math.log(i + 1, 2) for i in match])
            exp2dcg[exp[0]].append(len(match) / len(acus))

        ordered_by_len = sorted(exp2toks, key=lambda x: np.mean(exp2toks[x]))
        for k in ordered_by_len:
            # np.mean(horses[k])
            print(k, np.mean(exp2dcg[k]), np.mean(rouges[k]), np.mean(exp2toks[k]))
