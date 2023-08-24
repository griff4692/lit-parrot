import argparse

from datasets import load_dataset

from data_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--max_examples', default=50, type=int)

    args = parser.parse_args()

    others = ['length_test', 'length_w_dense_test']

    info = ['dense_gpt4_test']
    other_experiments = []
    experiment_fns = get_gpt4_fns(info)

    shared_ids = set([x[0] for x in experiment_fns])

    print('Reading in dataset...')
    dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')

    id2article = dict(zip(dataset['id'], dataset['article'] if 'article' in dataset.features else dataset['document']))
    id2reference = dict(
        zip(dataset['id'], dataset['highlights'] if 'highlights' in dataset.features else dataset['summary']))

    references = [
        id2reference[id] for id in shared_ids
    ]

    outputs = []
    for id in shared_ids:
        row = f'ID: {id}\n\n'

        baselines = [get_gpt4_preds([other], id, return_missing=False)[0] for other in others]
        dense, missing = get_gpt4_preds(info, id, return_missing=True)
        article = id2article[id]
        row += f'Article:\n{article}\n\n'
        for i, (d, m) in enumerate(zip(dense, missing)):
            row += f'Added Entity: {m}\nSummary {i + 1}: {d}\n\n'

        for pred, o in zip(baselines, others):
            row += f'Baseline {o}: {pred}\n\n'

        outputs.append(row)

    delim = '*' * 100 + '\n\n'
    with open('dense_examples.txt', 'w') as fd:
        fd.write(delim.join(outputs))
