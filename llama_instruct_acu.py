from collections import Counter, defaultdict
from datasets import Dataset, DatasetDict
import os
from glob import glob
from datasets import load_dataset
import argparse
import numpy as np
from oa_secrets import HF_ACCESS_TOKEN

from tqdm import tqdm
from autoacu import A3CU, A2CU


ALPACA_HEADER = 'Below is an instruction that describes a task, paired with an input that provides further context. ' \
                'Write a response that appropriately completes the request.'

INSTRUCTIONS = {
    'vanilla': 'Generate a concise and informative Summary of the Article.',
    'tldr': 'Generate a 1 sentence Summary of the Article.',
    'decide_length': 'Is this Summary too short? Answer Y/N.',
    'lengthen': 'Lengthen this Summary with important, non-redundant concepts in the Article.'
}


def score_acu(a3cu, prediction, reference):
    recall_scores, prec_scores, f1_scores = a3cu.score_example(
        candidate=prediction,
        reference=reference,
    )
    return {'recall': recall_scores, 'precision': prec_scores, 'f1': f1_scores}


def form(input, task_name):
    return f"{ALPACA_HEADER}\n\n### Instruction:\n{INSTRUCTIONS[task_name]}\n\n### Input:\n{input}\n\n### Response:\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--experiment', default='gpt-4_short_to_long_train')
    args = parser.parse_args()

    in_dir = os.path.join('out', 'adapter_v2', 's2l_llama_gpt4_selection', 'results', 'cnn', 'train')
    fns = list(glob(os.path.join(in_dir, '*.txt')))
    ids = list(set([x.split('/')[-1].replace('.txt', '') for x in fns]))

    dataset = load_dataset('cnn_dailymail', '3.0.0', split='train')
    id2reference = dict(zip(dataset['id'], dataset['highlights']))
    id2article = dict(zip(dataset['id'], dataset['article']))

    top_outputs = []
    outputs = {'train': []}

    a3cu = A3CU(device=1)  # the GPU device to use
    print(f'Loaded A3CU onto device...')

    dist = []
    max_f1s = []
    mean_f1s = []

    for fn in tqdm(fns, total=len(fns)):
        suffix = fn.split('/')[-1]
        id = suffix.replace('.txt', '').split('_')[0]
        # split = 'eval' if id in eval_ids else 'train'
        split = 'train'

        with open(fn, 'r') as fd:
            predictions = fd.readlines()
            article = id2article[id].strip()
            reference = id2reference[id]
            n = len(predictions)

            acus = [
                score_acu(a3cu, prediction, reference) for prediction in predictions
            ]

            acu_f1s = [x['f1'] for x in acus]
            max_f1 = max(acu_f1s)
            mean_f1 = np.mean(acu_f1s)
            max_f1s.append(max_f1)
            mean_f1s.append(mean_f1)
            print(f'Max ACU F1: {np.mean(max_f1s)}')
            print(f'Mean ACU F1: {np.mean(mean_f1s)}')

            top_idx = int(np.argmax(acu_f1s))

            first_summary = predictions[0]
            top_summary = predictions[top_idx]

            dist.append(top_idx + 1)
            print(Counter(dist).most_common())

            article_input = f'Article: {article}'
            outputs[split].append({
                'id': id,
                'prompt': form(article_input, 'vanilla'),
                'completion': top_summary,
                'task': 'summarize'
            })

    print('Building dataset from list...')
    outputs = DatasetDict({
        'train': Dataset.from_list(outputs['train']),
    })
    # print(f'Pushing {len(outputs)} examples to the Hub...')
    # outputs.push_to_hub('griffin/llama_summ', token=HF_ACCESS_TOKEN)
