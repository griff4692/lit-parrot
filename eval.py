from collections import defaultdict
import numpy as np
import os
from evaluate import load
from collections import Counter
from glob import glob

from datasets import load_dataset
from nltk import word_tokenize
import json
import openai
import backoff

from oa_secrets import OA_KEY, OA_ORGANIZATION

openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY


PREFIX = 'Here is an Article along with several possible summaries for the article.'
SUFFIXES = {
    'informative': 'Please rank the summaries from best to worst with respect to informativeness. An informative summary captures the important information in the article and presents it accurately and concisely. Return a JSON list of integers.',
    'quality': 'Please rank the summaries from best to worst with respect to quality. A high quality summary is comprehensible and understandable. Return a JSON list of integers.',
    'attributable': 'Please rank the summaries from best to worst with respect to attribution. Is all the information in the summary fully attributable to the Article? Return a JSON list of integers.',
    'concise': 'Please rank the summaries from best to worst with respect to conciseness. A concise summary gets straight to the point and does not include extraneous details. Return a JSON list of integers.',
}


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError), max_tries=25)
def chatgpt(messages, model='gpt-4', max_tokens=25):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', default='attributable')
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    overwrite = args.overwrite

    rouge = load('rouge', keep_in_memory=True)

    s2l_dir = os.path.expanduser('~/lit-parrot/out/adapter_v2/s2l_falcon/results')
    length_dir = os.path.expanduser('~/lit-parrot/out/adapter_v2/length_falcon/results')

    eval_dir = os.path.expanduser(f'~/lit-parrot/out/adapter_v2/{args.dimension}')
    os.makedirs(eval_dir, exist_ok=True)

    print('Reading in dataset...')
    if args.dataset == 'cnn':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
    else:
        dataset = load_dataset(args.dataset, split='test')

    id2article = dict(zip(dataset['id'], dataset['article']))
    id2reference = dict(zip(dataset['id'], dataset['highlights']))

    fns = list(glob(length_dir + '/*.txt'))
    ids_to_preds = defaultdict(dict)
    for fn in fns:
        id, task = fn.replace('.txt', '').split('/')[-1].split('_')
        with open(fn, 'r') as fd:
            pred = fd.read().strip()
            ids_to_preds[id][f'length_{task}'] = pred
    for id in ids_to_preds:
        s2l_fn = os.path.join(s2l_dir, f'{id}_summarize.txt')
        with open(s2l_fn, 'r') as fd:
            pred = fd.read().strip()
            ids_to_preds[id]['s2l'] = pred

    order = [
        's2l',
        'length_1',
        'length_2',
        'length_3',
        'length_4',
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

        if os.path.exists(out_fn) and not overwrite:
            with open(out_fn) as fd:
                ranking = json.load(fd)
            print(f'Loading from {out_fn}')
        else:
            article = id2article[id].strip()

            if len(article.split(' ')) > 2048:
                print(len(article.split(' ')))
                print('Truncating...')
                article = ' '.join(article.split(' ')[:2048])

            rand_order = np.arange(len(order))
            np.random.shuffle(rand_order)

            shuffled_predictions = [predictions[i] for i in rand_order]
            shuffled_order = [order[i] for i in rand_order]

            type_map = {i + 1: t for i, t in enumerate(shuffled_order)}

            pred_str = '\n\n'.join([
                f'Summary {i + 1}: {s}' for i, s in enumerate(shuffled_predictions)
            ])

            messages = [
                # Boost its ego first
                {'role': 'system', 'content': 'You are an evaluator of text summaries.'},
                {'role': 'user', 'content': f'{PREFIX}\n\nArticle: {article}\n\n{pred_str}\n\n{SUFFIXES[args.dimension]}'}
            ]

            output = chatgpt(messages=messages, model='gpt-4').strip()
            try:
                output = list(map(int, json.loads(output)))
            except:
                print(f'Could not parse output --> {output}')
                continue

            ranking = [type_map[i] for i in output]
            with open(out_fn, 'w') as fd:
                json.dump(ranking, fd, indent=4)
        for rank, type in enumerate(ranking):
            avg_rank[type].append(rank + 1)
        top_ranked.append(ranking[0])
        print(Counter(top_ranked).most_common())
        for type, arr in avg_rank.items():
            print(type + ' -> Rank ', str(np.mean(arr)))
        for type, arr in rouges.items():
            print(type + ' -> ROUGE1 ', str(np.mean(arr)))

    print('Tokens...')
    for task, cts in tokens.items():
        print(task + '\t' + str(np.mean(cts)))