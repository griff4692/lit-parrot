import os

from datasets import load_dataset, load_from_disk
from time import sleep
import backoff

from tqdm import tqdm
import numpy as np
import argparse
from oa_secrets import OA_KEY, OA_ORGANIZATION
import openai

openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY

from data_utils import *


PREFIX = 'Here is an Article along with several possible summaries for the article.'
SUFFIXES = {
    'informative': 'Please rate the summaries (1=worst to 5=best) with respect to informativeness. An informative summary captures the important information in the article and presents it accurately and concisely. Return a JSON list of integers.',
    'quality': 'Please rate the summaries (1=worst to 5=best) with respect to quality. A high quality summary is comprehensible and understandable. Return a JSON list of integers.',
    'attributable': 'Please rate the summaries (1=worst to 5=best) with respect to attribution. Is all the information in the summary fully attributable to the Article? Return a JSON list of integers.',
    'coherence': 'Please rate the summaries (1=worst to 5=best) with respect to coherence. A coherent summary is well-structured and well-organized. Return a JSON list of integers.',
    'overall': 'Please rate the summaries (1=worst to 5=best) with respect to overall preference. A good summary should convey the main ideas in the Article in a concise, logical, and coherent fashion. Return a JSON list of integers.',
    'detail': 'Please rate the summaries (1=worst to 5=best) with respect to the right level of detail. If the summary includes too many or too few details, give a low score. Return a JSON list of integers.',
    # 'balanced': 'Please rate the summaries (1=worst to 5=best) with respect to how well it how well it balances detail with readability. A well-balanced summary should include important details without being overly dense and hard to follow.\n\nPenalize overly dense summaries with awkward syntax.\n\nPenalize summaries which are not self-contained and can only be understand if supplied the Article.\n\nReturn a JSON list of integers.',
}


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError), max_tries=25)
def chatgpt(messages, model='gpt-4', max_tokens=32):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--dimension', default='informative')
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    EXPERIMENTS = GPT4_EXPERIMENTS
    get_fns = get_gpt4_fns
    get_preds = get_gpt4_preds
    split = 'test'

    experiment_fns = [
        get_fns(info) for info in EXPERIMENTS
    ]

    print([
        (EXPERIMENTS[i][0], len(experiment_fns[i])) for i in range(len(experiment_fns))
    ])

    shared_ids = set([x[0] for x in experiment_fns[0]])
    shared_ids = list(sorted(list(shared_ids)))

    eval_dir = os.path.expanduser(f'~/Desktop/s2l_data/cnn/dense_train_v2_{args.dimension}')
    os.makedirs(eval_dir, exist_ok=True)

    print('Reading in dataset...')
    if args.dataset == 'cnn':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split=split)
    elif args.dataset == 'xsum':
        dataset = load_dataset(args.dataset, split=split)
    else:
        dataset = load_from_disk(os.path.expanduser('~/nyt_edu_alignments'))[split]
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

    from collections import defaultdict
    all_likerts = defaultdict(list)
    NAMES = ['Initial', 'Step 1', 'Step 2', 'Step 3', 'Step 4']

    for id in tqdm(shared_ids):
        out_fn = os.path.join(eval_dir, f'{id}.json')
        if os.path.exists(out_fn) and not args.overwrite:
            print(f'Skipping {out_fn}')
            with open(out_fn, 'r') as fd:
                likerts = json.load(fd)
        else:
            preds = get_preds(EXPERIMENTS[0], id)
            pred_str = '\n\n'.join([
                f'Summary {i + 1}: {s}' for i, s in enumerate(preds)
            ])

            article = id2article[id]

            if args.dimension in {'quality', 'coherence'}:
                prompt = f'{PREFIX}\n\n{pred_str}\n\n{SUFFIXES[args.dimension]}'
            else:
                prompt = f'{PREFIX}\n\nArticle: {article}\n\n{pred_str}\n\n{SUFFIXES[args.dimension]}'

            messages = [
                # Boost its ego first
                {'role': 'system', 'content': 'You are an evaluator of text summaries.'},
                {'role': 'user', 'content': prompt}
            ]

            output = chatgpt(messages=messages, model='gpt-4')
            try:
                likerts = list(map(int, json.loads(output.split('\n')[0])))
            except:
                print(f'Could not parse output --> {output}')
                continue

            with open(out_fn, 'w') as fd:
                json.dump(likerts, fd)

            sleep(1)

        for likert, name in zip(likerts, NAMES):
            all_likerts[name].append(likert)
            print(name, np.mean(all_likerts[name]))
