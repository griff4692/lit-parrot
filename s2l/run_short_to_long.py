import os

import argparse
import json
from time import sleep

import openai
import numpy as np
from datasets import load_dataset
import ujson
from tqdm import tqdm
import backoff
import regex as re
import lxml.html.clean

from oa_secrets import OA_KEY, OA_ORGANIZATION

openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY


def clean_uuid(uuid):
    clean = re.sub(r'\W+', '_', uuid)
    return re.sub(r'_+', '_', clean).strip('_')


def linearize_chemistry(example):
    DELIM = '<!>'
    headers = example['headers'].split(DELIM)
    sections = example['sections'].split(DELIM)

    out_str = ''
    for header, body in zip(headers, sections):
        if header is not None and len(header.strip()) > 0:
            out_str += header.strip() + '\n\n'
        paragraphs = [x.strip() for x in re.split('</?p>', body) if len(x.strip()) > 0]
        out_str += '\n\n'.join(paragraphs)
        out_str += '\n\n'
    example['input'] = out_str.strip()
    return example


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError), max_tries=20)
def chatgpt(messages, model='gpt-4', max_tokens=1536):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.1, max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to Generate Two-Step Rank-then-Rate for Summeval')
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--mode', default='goyal', choices=[
        'short_to_long',
        'goyal',
    ])

    parser.add_argument('-entity_dense', default=False, action='store_true')

    parser.add_argument('--model', default='gpt-4', choices=[
        'gpt-3.5-turbo',
        'gpt-4',
    ])
    parser.add_argument('--split', default='train')

    parser.add_argument('--max_n', default=1000, type=int)

    parser.add_argument('-reverse', default=False, action='store_true')
    parser.add_argument('-shuffle', default=False, action='store_true')

    args = parser.parse_args()

    if args.dataset == 'chemistry':
        args.model = 'gpt-3.5-turbo-16k'

    if args.experiment is None:
        args.experiment = f'{args.model}_{args.mode}_{args.split}'

    if args.dataset == 'cnn':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split=args.split)
    elif args.dataset == 'fabbri':
        dataset = load_dataset('alexfabbri/answersumm', split=args.split)
    elif args.dataset == 'xsum':
        dataset = load_dataset('xsum', split=args.split)
    elif args.dataset == 'chemistry':
        dataset = load_dataset('griffin/chemsum', split=args.split)
    else:
        raise Exception('Unknown')

    n = len(dataset)
    if n > args.max_n:
        np.random.seed(1992)
        sample = list(sorted(np.random.choice(np.arange(n), size=(args.max_n, ), replace=False)))
        if args.reverse:
            sample = sample[::-1]
        elif args.shuffle:
            np.random.shuffle(sample)
        dataset = dataset.select(sample)

    out_dir = os.path.expanduser(os.path.join(f'~/Desktop/s2l_data/{args.dataset}/{args.experiment}'))
    print(f'Creating directory {out_dir} if does not exist...')
    os.makedirs(out_dir, exist_ok=True)

    source_key = 'Article' if args.dataset != 'chemistry' else 'Paper'

    s2l_prompt = open('s2l_prompt.txt').read().strip()

    arr = []

    for example in tqdm(dataset, total=len(dataset)):
        id = example.get('id', None)

        out_fn = os.path.join(out_dir, f'{id}.json')
        print(out_fn)
        if os.path.exists(out_fn):
            print(f'File exists. Skipping: {out_fn}')
            continue

        if id is None:
            id = clean_uuid(example.get('uuid'))
        if args.dataset == 'cnn':
            source = example['article'].strip()
            reference = example.get('reference', example['highlights'])
        elif args.dataset == 'chemistry':
            source = linearize_chemistry(example)['input']
            reference = example['abstract']
            doc = lxml.html.fromstring(source)
            cleaner = lxml.html.clean.Cleaner(style=True)
            doc = cleaner.clean_html(doc)
            source = doc.text_content()
        else:
            source = example['document'].strip()
            reference = example['summary']

        max_n = 2048 if args.dataset != 'chemistry' else 8192
        if len(source.split(' ')) > max_n:
            print(len(source.split(' ')))
            print('Truncating...')
            source = ' '.join(source.split(' ')[:max_n])

        if args.mode == 'goyal':
            suffix = '\n\n'
            if args.entity_dense:
                suffix = '\n\nThe summary should be entity dense.\n\n'

            initial_messages = [
                # Boost its ego first
                {'role': 'system', 'content': 'You are a helpful assistant for text summarization.'},
                # {'role': 'user', 'content': f'{source_key}: {source}\n\nSummarize the above {source_key} in {args.goyal_num} sentences.\n{suffix}'}
                {'role': 'user', 'content': f'{source_key}: {source}\n\nWrite a VERY short summary of the {source_key}.\n\nDo not exceed 70 words.\n\n{suffix}'.strip()}
            ]

            predictions = chatgpt(initial_messages, model=args.model)
            sleep(1)

            from nltk import word_tokenize
            arr.append(len(word_tokenize(predictions)))
            print(np.mean(arr))

            example['prediction'] = predictions
            example['mode'] = args.mode
            example['model'] = args.model
            example['experiment'] = args.experiment
            print(f'Saving to {out_fn}')
            with open(out_fn, 'w') as fd:
                ujson.dump(example, fd, indent=4)
        else:
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant for text summarization.'},
                {'role': 'user', 'content': s2l_prompt.replace('{{ARTICLE}}', source)}
            ]

            outputs = json.loads(chatgpt(messages))

            missing = [
                x["Missing_Entities"] for x in outputs
            ]

            predictions = [
                x["Longer_Summary"] for x in outputs
            ]

            example['prediction'] = predictions
            example['missing'] = missing
            example['mode'] = args.mode
            example['model'] = args.model
            example['experiment'] = args.experiment
            print(f'Saving to {out_fn}')
            with open(out_fn, 'w') as fd:
                ujson.dump(example, fd, indent=4)
