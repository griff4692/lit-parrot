from collections import defaultdict, Counter
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from datasets import load_dataset


SYSTEMS = [
    'Initial',
    'Step 1',
    'Step 2',
    'Step 3',
    'Step 4'
]


def rank_to_num(arr):
    ranks = [None for _ in range(len(arr))]
    for rank_idx, name in enumerate(arr):
        ranks[SYSTEMS.index(name)] = len(arr) - 1 - rank_idx
    return ranks


if __name__ == '__main__':
    metrics = json.load(open(os.path.expanduser('~/Desktop/s2l_data/metrics.json')))

    labels = pd.read_csv(os.path.expanduser('~/Desktop/s2l_data/Human_Eval_8_25.csv'))

    dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
    id2article = dict(zip(dataset['id'], dataset['article'] if 'article' in dataset.features else dataset['document']))

    id2source_toks = {}
    id2label = {}
    id2density_label = {}
    density_by_annotator = {'Griffin': [], 'Faisal': [], 'Alex': []}

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
        id2source_toks[row['id']] = len(id2article[row['id']].split(' '))

    print(pearsonr([id2source_toks[i] for i in id2source_toks], [id2density_label[i] for i in id2source_toks])[0])

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

        agg_dims = [sum([dim[i] for dim in dims]) / len(dims) for i in range(5)]
        print(agg_dims)
        if max(agg_dims) >= 4.5:
            intersection = {SYSTEMS[int(np.argmax(agg_dims))]}
        else:
            intersection = {}

        top_dims = [[i for i in range(len(x)) if x[i] == 5] for x in dims]
        # top_informative = [SYSTEMS[i] for i in top_dims[0]]
        # top_quality = [SYSTEMS[i] for i in top_dims[1]]
        # top_attributable = [SYSTEMS[i] for i in top_dims[2]]
        #
        # # top_agg = set(top_informative).intersection(set(top_quality)).intersection(set(top_attributable))
        # top_agg = set(top_quality).intersection(set(top_attributable))
        #
        # density = [metrics[sys]['num_ents_per_token'][idx] for sys in SYSTEMS]
        # ent_max_systems = [SYSTEMS[i] for i in range(len(density)) if density[i] >= 0.125]
        #
        # intersection = list(sorted(list(top_agg.intersection(set(ent_max_systems)))))

        for system in intersection:
            num_both += 1
            top_systems.append(system)
            if system in {SYSTEMS[i] for i in range(len(label)) if label[i] > 0}:
                agreement_both += 1

        def _agreement(a, b):
            a_systems = {SYSTEMS[i] for i in a}
            b_set = {SYSTEMS[i] for i in range(len(b)) if b[i] > 0}
            intersect = len(a_systems.intersection(b_set))
            return intersect / (0.5 * len(a_systems) + 0.5 * len(b_set))

        agreements = [
            _agreement(a, label) for a in top_dims
        ]

        # ensemble = [a * b for a, b in zip(dim_nums[0], dim_nums[1])]
        # instances['gpt_ensemble'].append(float(pearsonr(ensemble, label)[0]))

        for dim_name, dim_vals, agreement in zip(dimensions, dims, agreements):
            instances[dim_name].append(float(pearsonr(dim_vals, label)[0]))
            instances[dim_name + '_agreement'].append(agreement)

        instances['gpt4_ner'].append(float(pearsonr([len(x) for x in gpt_ent], label)[0]))
        for col in cols:
            system_outs = [metrics[sys][col][idx] for sys in SYSTEMS]
            corel = float(pearsonr(system_outs, label)[0])
            instances[col].append(corel)
    for k, v in instances.items():
        print(k, np.mean([p for p in v if not np.isnan(p)]))

    print(agreement_both, num_both, None if num_both == 0 else round(agreement_both / num_both, 2))
    print(Counter(top_systems).most_common())
