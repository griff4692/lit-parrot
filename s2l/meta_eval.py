from collections import defaultdict
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


SYSTEMS = [
    'Initial',
    'Step 1',
    'Step 2',
    'Step 3'
]


def rank_to_num(arr):
    ranks = [None for _ in range(len(arr))]
    for rank_idx, name in enumerate(arr):
        ranks[SYSTEMS.index(name)] = len(arr) - 1 - rank_idx
    return ranks


if __name__ == '__main__':
    metrics = json.load(open(os.path.expanduser('~/Desktop/s2l_data/metrics.json')))

    labels = pd.read_csv(os.path.expanduser('~/Desktop/s2l_data/cnn/Human_Eval_8_20.csv'))

    id2label = {}
    for row in labels.to_dict('records'):
        l1 = row['Griffin Mapped']
        l2 = row['Alex Mapped']
        l_agg = [0, 0, 0, 0]
        l_agg[SYSTEMS.index(l1)] += 1
        l_agg[SYSTEMS.index(l2)] += 1
        l_agg = [x / 2.0 for x in l_agg]

        id2label[row['id']] = l_agg

    dimensions = [
        'informative',
        'quality',
        'overall',
        'coherence',
        'attributable',
    ]
    dim_dirs = [
        os.path.expanduser(f'~/Desktop/s2l_data/human_dense_{dimension}') for dimension in dimensions
    ]

    # Instance Level
    instances = defaultdict(list)
    cols = [k for k, v in metrics['Initial'].items() if type(v) == list]

    agreement_both = 0
    num_both = 0
    top_systems = []

    for idx, id in enumerate(metrics['id']):
        label = id2label[id]
        entity_fn = os.path.expanduser(f'~/Desktop/s2l_data/human_dense_entity/{id}.json')
        with open(entity_fn, 'r') as fd:
            gpt_ent = json.load(fd)

        dims = [
            json.load(open(os.path.join(dim_dir, f'{id}.json'), 'r')) for dim_dir in dim_dirs
        ]

        top_dims = [x[0] for x in dims]

        num_ents = [metrics[sys]['num_ents'][idx] for sys in SYSTEMS]
        ent_max_systems = [SYSTEMS[i] for i in range(len(num_ents)) if num_ents[i] == max(num_ents)]

        # if top_dims[0] == top_dims[1]:
        if top_dims[1] in ent_max_systems:
            num_both += 1
            top_systems.append(top_dims[1])
            if top_dims[1] in {SYSTEMS[i] for i in range(len(label)) if label[i] > 0}:
                agreement_both += 1

        dim_nums = [
            rank_to_num(dim) for dim in dims
        ]

        def _agreement(a, b):
            b_set = {SYSTEMS[i] for i in range(len(b)) if b[i] > 0}
            return 1 if a in b_set else 0

        agreements = [
            _agreement(a, label) for a in top_dims
        ]

        # ensemble = [a * b for a, b in zip(dim_nums[0], dim_nums[1])]
        # instances['gpt_ensemble'].append(float(pearsonr(ensemble, label)[0]))

        for dim_name, dim_vals, agreement in zip(dimensions, dim_nums, agreements):
            instances[dim_name].append(float(pearsonr(dim_vals, label)[0]))
            instances[dim_name + '_agreement'].append(agreement)

        instances['gpt4_ner'].append(float(pearsonr([len(x) for x in gpt_ent], label)[0]))
        for col in cols:
            system_outs = [metrics[sys][col][idx] for sys in SYSTEMS]
            corel = float(pearsonr(system_outs, label)[0])
            instances[col].append(corel)
    for k, v in instances.items():
        print(k, np.mean([p for p in v if not np.isnan(p)]))

    print(agreement_both, num_both)
    from collections import Counter
    print(Counter(top_systems).most_common())
