from collections import defaultdict
from datasets import Dataset, DatasetDict
import os
from glob import glob
import argparse
from oa_secrets import HF_ACCESS_TOKEN
import numpy as np
import ujson
from tqdm import tqdm

ALPACA_HEADER = 'Below is an instruction that describes a task, paired with an input that provides further context. ' \
                'Write a response that appropriately completes the request.'

INSTRUCTIONS = {
    'straight': 'Generate an entity-dense summary of the Article.',
}


NAMES = ['Initial', 'Step 1', 'Step 2', 'Step 3']


def form(input, task_name='straight'):
    return f"{ALPACA_HEADER}\n\n### Instruction:\n{INSTRUCTIONS[task_name]}\n\n### Input:\n{input}\n\n### Response:\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--experiment', default='dense_train')
    args = parser.parse_args()

    in_dir = os.path.expanduser(f'~/Desktop/s2l_data/cnn/{args.experiment}')
    fns = list(glob(os.path.join(in_dir, '*.json')))
    ids = list(set([x.split('/')[-1].replace('.json', '') for x in fns]))

    stats_fn = os.path.expanduser('~/Desktop/s2l_data/cnn/dense_train_stats.json')
    quality_dir = os.path.expanduser('~/Desktop/s2l_data/cnn/dense_train_quality')
    with open(stats_fn, 'r') as fd:
        stats = ujson.load(fd)

    id2stats = {}
    for idx, id in enumerate(stats['id']):
        id2stats[id] = [stats[name]['num_ents'][idx] for name in NAMES]

    top_outputs = []
    outputs = {'train': []}

    dist = []
    max_f1s = []

    by_step = [
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    ]

    cts = []
    for fn in tqdm(fns, total=len(fns)):
        suffix = fn.split('/')[-1]
        id = suffix.replace('.json', '')
        # split = 'eval' if id in eval_ids else 'train'
        split = 'train'

        with open(fn, 'r') as fd:
            result = ujson.load(fd)
            article = result['article'].strip()
            reference = result['highlights'].strip()
            predictions = result['prediction'][:4]
            n = len(predictions)

            num_ents = id2stats[id]
            quality_fn = os.path.join(quality_dir, f'{id}.json')
            with open(quality_fn, 'r') as fd:
                top_quality = ujson.load(fd)[0]

            ent_max_systems = [NAMES[i] for i in range(len(num_ents)) if num_ents[i] == max(num_ents)]

            if top_quality not in ent_max_systems:
                continue

            top_summary = predictions[NAMES.index(top_quality)]

            outputs[split].append({
                'id': id,
                'prompt': form(article),
                'completion': top_summary,
                'step': top_quality,
            })

            cts.append(top_quality)

    from collections import Counter
    print(Counter(cts).most_common())
    print('Building dataset from list...')
    outputs = DatasetDict({
        'train': Dataset.from_list(outputs['train']),
    })
    n_out = len(outputs['train'])
    print(f'Pushing {n_out} examples to the Hub...')
    outputs.push_to_hub('griffin/dense_summ', token=HF_ACCESS_TOKEN)

