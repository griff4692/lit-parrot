import argparse

from datasets import load_dataset
import numpy as np
import pandas as pd

from data_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--max_examples', default=50, type=int)

    args = parser.parse_args()

    experiment_fns = [
        get_gpt4_fns(info) for info in GPT4_EXPERIMENTS
    ]

    print([
        (GPT4_EXPERIMENTS[i][0], len(experiment_fns[i])) for i in range(len(experiment_fns))
    ])

    shared_ids = set([x[0] for x in experiment_fns[0]])
    for i in range(1, len(experiment_fns)):
        shared_ids.intersection(set([x[0] for x in experiment_fns[i]]))

    shared_ids = list(sorted(list(shared_ids)))
    assert len(shared_ids) >= args.max_examples

    if len(shared_ids) > args.max_examples:
        np.random.seed(1992)
        np.random.shuffle(shared_ids)
        shared_ids = shared_ids[:args.max_examples]

    print('Reading in dataset...')
    dataset = load_dataset('cnn_dailymail', '3.0.0', split='train')

    id2article = dict(zip(dataset['id'], dataset['article'] if 'article' in dataset.features else dataset['document']))
    id2reference = dict(
        zip(dataset['id'], dataset['highlights'] if 'highlights' in dataset.features else dataset['summary']))

    references = [
        id2reference[id] for id in shared_ids
    ]

    outputs = []
    meta = []
    from nltk import word_tokenize

    tokens = [
        [], [], [], [], [],
    ]
    for id in shared_ids:
        row = f'ID: {id}\n\n'
        all_s2l = get_gpt4_preds(GPT4_EXPERIMENTS[0], id)
        compare_idx = int(np.random.randint(1, len(GPT4_EXPERIMENTS)))
        compare_exp = GPT4_EXPERIMENTS[compare_idx]
        compare_pred = get_gpt4_preds(compare_exp, id)[0].strip()

        for i in range(len(all_s2l)):
            tokens[i].append(len(word_tokenize(all_s2l[i])))
            print(i, np.mean(tokens[i]))

        chosen_s2l_idx = int(np.random.randint(1, len(all_s2l) - 1))
        s2l = all_s2l[chosen_s2l_idx]

        article = id2article[id]
        s2l_first = np.random.random() > 0.5
        row += f'Article:\n{article}\n\n'
        if s2l_first:
            row += f'Summary A: {s2l}\n\nSummary B: {compare_pred}\n\n'
        else:
            row += f'Summary A: {compare_pred}\n\nSummary B: {s2l}\n\n'

        row += 'Alex Preference:\n\nGriffin Preference:\n\nFaisal Preference:\n\n'
        outputs.append(row)
        meta.append({
            'comparison': compare_exp[0],
            'chosen_s2l_idx': chosen_s2l_idx,
            's2l_is_summary_a': s2l_first,
        })

    meta = pd.DataFrame(meta)
    delim = '*' * 100 + '\n\n'
    with open('human_eval.txt', 'w') as fd:
        fd.write(delim.join(outputs))

    meta.to_csv('human_meta.csv', index=False)
