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
    'straight': 'Generate a summary with the top {{K}} most salient entities.',
    'straight_w_plan': 'Identify the top {{K}} most salient entities and then generate a summary.',
}


def form(input, task_name='straight', K=None):
    prompt = f"{ALPACA_HEADER}\n\n### Instruction:\n{INSTRUCTIONS[task_name]}\n\n### Input:\n{input}\n\n### Response:\n"
    prompt = prompt.replace('{{K}}', str(K))
    return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--experiment', default='train_straight_dense')
    args = parser.parse_args()

    in_dir = os.path.expanduser(f'~/Desktop/s2l_data/cnn/{args.experiment}')
    fns = list(glob(os.path.join(in_dir, '*.json')))
    ids = list(set([x.split('/')[-1].replace('.json', '') for x in fns]))

    outputs = {'train': []}

    for fn in tqdm(fns, total=len(fns)):
        suffix = fn.split('/')[-1]
        id = suffix.replace('.json', '')
        # split = 'eval' if id in eval_ids else 'train'
        split = 'train'

        with open(fn, 'r') as fd:
            result = ujson.load(fd)
            article = result['article'].strip()
            reference = result['highlights'].strip()
            prediction = result['prediction']
            K = len(result['entities'])
            entities = "; ".join(result['entities'])

            article_str = f'### Article:\n{article}'

            outputs[split].append({
                'id': id,
                'task': 'straight',
                'prompt': form(article_str, K=K, task_name='straight'),
                'completion': f'SUMMARY: {prediction}',
            })

            outputs[split].append({
                'id': id,
                'task': 'straight_w_plan',
                'prompt': form(article_str, K=K, task_name='straight_w_plan'),
                'completion': f'ENTITIES: {entities}\n\nSUMMARY: {prediction}',
            })

    print('Building dataset from list...')
    outputs = DatasetDict({
        'train': Dataset.from_list(outputs['train']),
    })
    n_out = len(outputs['train'])
    print(f'Pushing {n_out} examples to the Hub...')
    outputs.push_to_hub('griffin/straight_dense_summ', token=HF_ACCESS_TOKEN)
