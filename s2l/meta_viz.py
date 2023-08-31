import json
import os

import pandas as pd
from datasets import load_dataset


from data_utils import get_gpt4_fns, get_gpt4_preds

SYSTEMS = [
    'Initial',
    'Step 1',
    'Step 2',
    'Step 3',
    'Step 4'
]


if __name__ == '__main__':
    metrics = json.load(open(os.path.expanduser('~/Desktop/s2l_data/metrics.json')))

    labels = pd.read_csv(os.path.expanduser('~/Desktop/s2l_data/Human_Eval_8_25.csv'))

    dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
    id2article = dict(zip(dataset['id'], dataset['article'] if 'article' in dataset.features else dataset['document']))

    id2label = {}
    id2density_label = {}
    density_by_annotator = {'Griffin': [], 'Faisal': [], 'Alex': []}

    info = ['human_dense_v3']
    pred_fns = get_gpt4_fns(info)
    id2preds = {}
    for id, fn in pred_fns:
        id2preds[id] = get_gpt4_preds(info, id)

    id2annotations = {}
    for row in labels.to_dict('records')[:100]:
        l1 = row['Griffin Mapped']
        l2 = row['Faisal Mapped']
        l3 = row['Alex Mapped']
        l_agg = [0, 0, 0, 0, 0]
        l_agg[SYSTEMS.index(l1)] += 1
        l_agg[SYSTEMS.index(l2)] += 1
        l_agg[SYSTEMS.index(l3)] += 1
        l_agg = [x / 3.0 for x in l_agg]
        id2label[row['id']] = l_agg
        id2density_label[row['id']] = sum([SYSTEMS.index(l1), SYSTEMS.index(l2), SYSTEMS.index(l3)]) / 3.0
        id2annotations[row['id']] = {'Griffin': l1, 'Faisal': l2, 'Alex': l3}

    dimensions = [
        'informative',
        'quality',
        'attributable',
        'overall',
        'coherence',
        'detail',
    ]
    dim_dirs = [
        os.path.expanduser(f'~/Desktop/s2l_data/cnn/human_dense_v3_{dimension}') for dimension in dimensions
    ]

    outputs = []

    for idx, id in enumerate(metrics['id']):
        label = id2label[id]
        entity_fn = os.path.expanduser(f'~/Desktop/s2l_data/human_dense_entity/{id}.json')
        with open(entity_fn, 'r') as fd:
            gpt_ent = json.load(fd)

        dims = [
            json.load(open(os.path.join(dim_dir, f'{id}.json'), 'r')) for dim_dir in dim_dirs
        ]

        article = id2article[id]

        row = f'ID: {id}\n\n'
        row += f'Article:\n{article}\n\n'
        for j, name in enumerate(SYSTEMS):
            row += f'{name}: {id2preds[id][j]}\n\n'

        for dim, vals in zip(dimensions, dims):
            row += f'GPT-4 Scores for {dim}:\n'
            for j, name in enumerate(SYSTEMS):
                row += f'- {name}: {vals[j]}\n'
            row += '\n'

        row += "Entities / Token:\n"
        for name in SYSTEMS:
            row += f"{name}: {round(metrics[name]['num_ents_per_token'][idx], 3)}\n"

        row += "\nHUMANS:\n"
        row += f"- Griffin: {id2annotations[id]['Griffin']}\n"
        row += f"- Alex: {id2annotations[id]['Alex']}\n"
        row += f"- Faisal: {id2annotations[id]['Faisal']}\n"

        row += '\n'
        outputs.append(row)

    DELIM = '*' * 75 + '\n\n'
    with open('meta_viz.txt', 'w') as fd:
        fd.write(DELIM.join(outputs))
