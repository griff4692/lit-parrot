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


PROMPT = """
Article: {{ARTICLE}}

Summary: {{SUMMARY}}

List out the unique named entities in the Summary.
- Write out full proper nouns, not acronyms, abbreviations, or pronouns
- Break up compound entities: e.g., "New York Mayor Rudy Giuliani" into "New York Mayor" and "Rudy Giuliani"
- Include each unique named entity once (only the first time it is mentioned)

Return a valid python list of strings ([] and double quotes "").
"""


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError), max_tries=20)
def chatgpt(messages, model='gpt-4', max_tokens=200):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--max_examples', default=100, type=int)

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
    assert len(shared_ids) >= args.max_examples

    entity_dir = os.path.expanduser(f'~/Desktop/s2l_data/human_dense_entity')
    os.makedirs(entity_dir, exist_ok=True)

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

    output = {}
    nums = [[], [], [], []]

    for id in tqdm(shared_ids):
        out_fn = os.path.join(entity_dir, f'{id}.json')
        if os.path.exists(out_fn):
            print(f'Skipping {out_fn}')
            with open(out_fn, 'r') as fd:
                row = json.load(fd)
        else:
            preds = get_preds(EXPERIMENTS[0], id)
            row = []
            for step in range(4):
                prompt = PROMPT.replace("{{ARTICLE}}", id2article[id])
                prompt = prompt.replace("{{SUMMARY}}", preds[step])
                initial_messages = [
                    # Boost its ego first
                    {'role': 'system', 'content': 'You are a precise extractor of entities.'},
                    {'role': 'user', 'content': prompt}
                ]

                ents = ast.literal_eval(chatgpt(initial_messages))
                row.append(ents)
            with open(out_fn, 'w') as fd:
                json.dump(row, fd)

        for step in range(4):
            nums[step].append(len(row[step]))
            print(step, np.mean(nums[step]))
