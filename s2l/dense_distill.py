from datasets import Dataset, DatasetDict
import os
from glob import glob
import argparse
from oa_secrets import HF_ACCESS_TOKEN
import ujson
from tqdm import tqdm

ALPACA_HEADER = 'Below is an instruction that describes a task, paired with an input that provides further context. ' \
                'Write a response that appropriately completes the request.'

INSTRUCTIONS = {
    'straight': 'Generate an entity-dense summary.',
    'densify': 'Incorporate 1-3 new entities into an existing summary.',
}


NAMES = ['Initial', 'Step 1', 'Step 2', 'Step 3', 'Step 4']


def form(input, task_name='straight'):
    return f"{ALPACA_HEADER}\n\n### Instruction:\n{INSTRUCTIONS[task_name]}\n\n### Input:\n{input}\n\n### Response:\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--experiment', default='dense_train_v2')
    args = parser.parse_args()

    in_dir = os.path.expanduser(f'~/Desktop/s2l_data/cnn/{args.experiment}')
    fns = list(glob(os.path.join(in_dir, '*.json')))
    ids = list(set([x.split('/')[-1].replace('.json', '') for x in fns]))

    outputs = {'train': []}

    SYSTEMS = ['Initial', 'Step 1', 'Step 2', 'Step 3', 'Step 4']

    unique_ids = set()
    for fn in tqdm(fns, total=len(fns)):
        suffix = fn.split('/')[-1]
        id = suffix.replace('.json', '')
        # split = 'eval' if id in eval_ids else 'train'
        split = 'train'

        with open(fn, 'r') as fd:
            result = ujson.load(fd)
            article = result['article'].strip()
            reference = result['highlights'].strip()
            predictions = result['prediction']
            missing = result['missing']
            n = len(predictions)

            for j in range(5):
                prev_summary = '' if j == 0 else predictions[j - 1]

                article_str = f'### Article:\n{article}'
                prev_summary_str = f'### Existing Summary:\n{prev_summary}'
                straight_input = article_str
                densify_input = f'{article_str}\n\n{prev_summary_str}'

                outputs[split].append({
                    'id': id,
                    'task': 'straight',
                    'step': SYSTEMS[j],
                    'prompt': form(straight_input, task_name='straight'),
                    'completion': predictions[j],
                })

                outputs[split].append({
                    'id': id,
                    'task': 'densify',
                    'step': SYSTEMS[j],
                    'prompt': form(densify_input, task_name='densify'),
                    'completion': f'ENTITIES: {missing[j]}\n\nSUMMARY: {predictions[j]}',
                })

    print(f'Unique IDS: {len(unique_ids)}')
    print('Building dataset from list...')
    outputs = DatasetDict({
        'train': Dataset.from_list(outputs['train']),
    })
    n_out = len(outputs['train'])
    print(f'Pushing {n_out} examples to the Hub...')
    outputs.push_to_hub('griffin/dense_summ_v2', token=HF_ACCESS_TOKEN)
