from collections import defaultdict
import numpy as np
import os
from evaluate import load
from time import sleep
from collections import Counter
from glob import glob
from tqdm import tqdm

from datasets import load_dataset
from nltk import word_tokenize
import json
import openai
import backoff

from oa_secrets import OA_KEY, OA_ORGANIZATION

openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY


EXPERIMENTS = [
    # ['llama_chat_incr', 's2l_llama_chat', 's2l'],
    # ['llama_chat_straight', 's2l_llama_chat', 'summarize'],
    ['llama_incr', 's2l_llama', 's2l'],
    ['llama_straight', 's2l_llama', 'summarize'],
    # ['falcon_incr', 's2l_falcon', 's2l'],
    # ['falcon_straight', 's2l_falcon', 'summarize'],
    # ['llama_chat_1', 'length_llama_chat', '1'],
    # ['llama_chat_2', 'length_llama_chat', '2'],
    # ['llama_chat_3', 'length_llama_chat', '3'],
    # ['llama_chat_4', 'length_llama_chat', '4'],
    ['llama_1', 'length_llama', '1'],
    ['llama_2', 'length_llama', '2'],
    ['llama_3', 'length_llama', '3'],
    ['llama_4', 'length_llama', '4'],
    # ['falcon_1', 'length_falcon', '1'],
    # ['falcon_2', 'length_falcon', '2'],
    # ['falcon_3', 'length_falcon', '3'],
    # ['falcon_4', 'length_falcon', '4'],
]


PREFIX = 'Here is an Article along with several possible summaries for the article.'
SUFFIXES = {
    'informative': 'Please rank the summaries from best to worst with respect to informativeness. An informative summary captures the important information in the article and presents it accurately and concisely. Return a JSON list of integers.',
    'quality': 'Please rank the summaries from best to worst with respect to quality. A high quality summary is comprehensible and understandable. Return a JSON list of integers.',
    'attributable': 'Please rank the summaries from best to worst with respect to attribution. Is all the information in the summary fully attributable to the Article? Return a JSON list of integers.',
    'concise': 'Please rank the summaries from best to worst with respect to conciseness. A concise summary gets straight to the point and does not include extraneous details. Return a JSON list of integers.',
    'redundancy': 'Please rank the summaries from best to worst with respect to non-redundancy. A non-redundant summary contains does not repeat a fact more than once. Return a JSON list of integers.',
}


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError), max_tries=25)
def chatgpt(messages, model='gpt-4', max_tokens=64):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', default='informative')
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--experiment', default='default')
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--max_examples', default=25, type=int)

    args = parser.parse_args()

    overwrite = args.overwrite

    rouge = load('rouge', keep_in_memory=True)

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

    eval_dir = os.path.expanduser(f'~/lit-parrot/out/eval/{args.experiment}/{args.dimension}')
    os.makedirs(eval_dir, exist_ok=True)

    print('Reading in dataset...')
    if args.dataset == 'cnn':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
    else:
        dataset = load_dataset(args.dataset, split='test')

    id2article = dict(zip(dataset['id'], dataset['article']))
    id2reference = dict(zip(dataset['id'], dataset['highlights']))

    top_ranked = []

    avg_rank = {}
    tokens = {}
    rouges = {}
    for x in EXPERIMENTS:
        tokens[x[0]] = []
        avg_rank[x[0]] = []
        rouges[x[0]] = []

    for id in tqdm(shared_ids):
        out_fn = os.path.join(eval_dir, f'{id}.json')
        reference = id2reference[id]

        predictions = [
            get_pred(info, id) for info in EXPERIMENTS
        ]

        for pred, type in zip(predictions, EXPERIMENTS):
            r1 = rouge.compute(predictions=[pred], references=[reference], rouge_types=['rouge1'])['rouge1']
            rouges[type[0]].append(r1)

        token_cts = [
            len(word_tokenize(x)) for x in predictions
        ]

        for task, ct in zip(EXPERIMENTS, token_cts):
            tokens[task[0]].append(ct)

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

            rand_order = np.arange(len(EXPERIMENTS))
            np.random.shuffle(rand_order)

            shuffled_predictions = [predictions[i] for i in rand_order]
            shuffled_order = [EXPERIMENTS[i][0] for i in rand_order]

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
            sleep(3)
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

    print('Final Tokens and ROUGES...')
    for task, cts in tokens.items():
        print(task + '\t' + str(np.mean(cts)) + '\t' + str(np.mean(rouges[task])))

    print('Tokens and Rankings...')
    for task, cts in tokens.items():
        print(task + '\t' + str(np.mean(cts)) + '\t' + str(np.mean(avg_rank[task])))
