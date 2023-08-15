from collections import Counter, defaultdict
from datasets import Dataset, DatasetDict
import os
from glob import glob
import argparse
from oa_secrets import HF_ACCESS_TOKEN

import ujson
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
        reference=reference,
        candidate=prediction,
    )
    return {'recall': recall_scores, 'precision': prec_scores, 'f1': f1_scores}


def form(input, task_name):
    return f"{ALPACA_HEADER}\n\n### Instruction:\n{INSTRUCTIONS[task_name]}\n\n### Input:\n{input}\n\n### Response:\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--experiment', default='gpt-4_short_to_long_train')
    args = parser.parse_args()

    in_dir = f'data/{args.experiment}'
    fns = list(glob(os.path.join(in_dir, '*.json')))
    ids = list(set([x.split('/')[-1].replace('.json', '') for x in fns]))

    import numpy as np

    top_outputs = []
    outputs = {'train': []}

    a3cu = A3CU(device=0)  # the GPU device to use
    print(f'Loaded A3CU onto device...')

    dist = []
    max_f1s = []

    by_step = [
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    ]

    confusion_matrix = np.zeros([4, 4])

    for fn in tqdm(fns, total=len(fns)):
        suffix = fn.split('/')[-1]
        id = suffix.replace('.json', '')
        # split = 'eval' if id in eval_ids else 'train'
        split = 'train'

        eval_fn = f'data/{args.experiment}_eval/{id}.txt'
        with open(eval_fn, 'r') as fd:
            gpt_selection_idx = int(fd.read().strip()) - 1
            assert gpt_selection_idx >= 0

        with open(fn, 'r') as fd:
            result = ujson.load(fd)
            article = result['article'].strip()
            reference = result['highlights'].strip()
            predictions = result['prediction']
            missing = result['missing']
            n = len(predictions)

            acus = [
                score_acu(a3cu, prediction, reference) for prediction in predictions
            ]

            acu_f1s = [x['f1'] for x in acus]
            max_f1 = max(acu_f1s)
            max_f1s.append(max_f1)
            print(f'Max ACU F1: {np.mean(max_f1s)}')

            for step, acu in enumerate(acus):
                by_step[step]['precision'].append(acu['precision'])
                by_step[step]['recall'].append(acu['recall'])
                by_step[step]['f1'].append(acu['f1'])
                print(
                    f"Step {step + 1}: {np.mean(by_step[step]['precision'])}, "
                    f"{np.mean(by_step[step]['recall'])}, {np.mean(by_step[step]['f1'])}"
                )

            top_idx = int(np.argmax(acu_f1s))

            confusion_matrix[gpt_selection_idx, top_idx] += 1
            print(confusion_matrix)

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

            tldr_completion = first_summary
            outputs[split].append({
                'id': id,
                'prompt': form(article_input, 'tldr'),
                'completion': tldr_completion,
                'task': 'tldr'
            })

            decide_input = f'Article: {article}\n\nSummary: {top_summary}'
            outputs[split].append({
                'id': id,
                'prompt': form(decide_input, 'decide_length'),
                'completion': 'N',
                'task': 'decide_length',
            })

            prev_idxs = list(range(top_idx))
            for prev_idx in prev_idxs:
                decide_input = f'Article: {article}\n\nSummary: {predictions[prev_idx]}'
                outputs[split].append({
                    'id': id,
                    'prompt': form(decide_input, 'decide_length'),
                    'completion': 'Y',
                    'task': 'decide_length'
                })

            adjacent_idx = top_idx - 1
            if adjacent_idx >= 0:
                add = '; '.join(missing[adjacent_idx])
                lengthen_input = f'Article: {article}\n\nSummary: {predictions[adjacent_idx]}'
                lengthen_completion = f'MISSING: {add}\nSUMMARY V2: {top_summary}'
                outputs[split].append({
                    'id': id,
                    'prompt': form(lengthen_input, 'lengthen'),
                    'completion': lengthen_completion,
                    'task': 'change_length'
                })

    print('Building dataset from list...')
    outputs = DatasetDict({
        'train': Dataset.from_list(outputs['train']),
    })
    # print(f'Pushing {len(outputs)} examples to the Hub...')
    # outputs.push_to_hub('griffin/incr_summ', token=HF_ACCESS_TOKEN)
