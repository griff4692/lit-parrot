import os

from datasets import load_dataset, load_from_disk
import ast
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
    'informative': 'Please rank the summaries from best to worst with respect to informativeness. An informative summary captures the important information in the article and presents it accurately and concisely. Return a JSON list of integers.',
    'quality': 'Please rank the summaries from best to worst with respect to quality. A high quality summary is comprehensible and understandable. Return a JSON list of integers.',
    'attributable': 'Please rank the summaries from best to worst with respect to attribution. Is all the information in the summary fully attributable to the Article? Return a JSON list of integers.',
    'coherence': 'Please rank the summaries from best to worst with respect to coherence. A coherent summary is well-structured and well-organized. Return a JSON list of integers.',
    'overall': 'Please rank the summaries from best to worst with respect to overall preference. A good summary should convey the main ideas in the Article in a concise, logical, and coherent fashion. Return a JSON list of integers.',
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
    split = 'train'

    experiment_fns = [
        get_fns(info) for info in EXPERIMENTS
    ]

    print([
        (EXPERIMENTS[i][0], len(experiment_fns[i])) for i in range(len(experiment_fns))
    ])

    shared_ids = set([x[0] for x in experiment_fns[0]])
    shared_ids = list(sorted(list(shared_ids)))

    eval_dir = os.path.expanduser(f'~/Desktop/s2l_data/cnn/dense_train_{args.dimension}')
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
    rankings = defaultdict(list)
    NAMES = ['Initial', 'Step 1', 'Step 2', 'Step 3', 'Step 4']

    for id in tqdm(shared_ids):
        out_fn = os.path.join(eval_dir, f'{id}.json')
        if os.path.exists(out_fn):
            print(f'Skipping {out_fn}')
            with open(out_fn, 'r') as fd:
                ranking = json.load(fd)
        else:
            preds = get_preds(EXPERIMENTS[0], id)[:4]
            rand_order = np.arange(len(preds))
            np.random.shuffle(rand_order)

            preds_rand = [preds[i] for i in rand_order]
            experiments_rand = [NAMES[i] for i in rand_order]

            type_map = {i + 1: t for i, t in enumerate(experiments_rand)}

            pred_str = '\n\n'.join([
                f'Summary {i + 1}: {s}' for i, s in enumerate(preds_rand)
            ])

            article = id2article[id]

            prompt = f'{PREFIX}\n\nArticle: {article}\n\n{pred_str}\n\n{SUFFIXES[args.dimension]}'

            messages = [
                # Boost its ego first
                {'role': 'system', 'content': 'You are an evaluator of text summaries.'},
                {'role': 'user', 'content': prompt}
            ]

            output = chatgpt(messages=messages, model='gpt-4')
            try:
                output = list(map(int, json.loads(output.split('\n')[0])))
                ranking = [type_map[i] for i in output]
            except:
                print(f'Could not parse output --> {output}')
                continue

            with open(out_fn, 'w') as fd:
                json.dump(ranking, fd)

        for rank_idx, name in enumerate(ranking):
            rankings[name].append(rank_idx + 1)
            print(name, np.mean(rankings[name]))
