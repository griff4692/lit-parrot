import os
import json
from glob import glob


LLAMA_EXPERIMENTS = [
    ['llama_straight', 's2l_llama', 'summarize'],
    ['llama_1', 'length_llama', '1'],
    ['llama_2', 'length_llama', '2'],
    ['llama_3', 'length_llama', '3'],
    ['llama_4', 'length_llama', '4'],
]


GPT4_EXPERIMENTS = [
    ['human_dense_v2'],
    # ['gpt4_length', 'length_test'],
    # ['gpt4_length_w_dense', 'length_w_dense_test'],
]

GPT4_DIR = os.path.expanduser(os.path.join('~', 'Desktop', 's2l_data', 'cnn'))


def get_gpt4_preds(info, id, return_missing=False, dataset=None):
    suffix = info[-1]
    fn = os.path.join(GPT4_DIR, suffix, f'{id}.json')
    fn = os.path.expanduser(fn)
    with open(fn, 'r') as fd:
        preds = json.load(fd)
    predictions = preds['prediction']
    if type(predictions) == str:
        return [predictions]
    if return_missing:
        return predictions, preds['missing']
    return predictions


def get_gpt4_fns(info, dataset=None):
    suffix = info[-1]
    pattern = os.path.join(GPT4_DIR, suffix, '*.json')
    fns = list(glob(pattern))
    ids = [
        fn.split('/')[-1].replace('.json', '').split('_')[0] for fn in fns
    ]
    return list(zip(ids, fns))


def get_llama_preds(info, id, dataset):
    suffix = info[-1]
    fn = 'out/adapter_v2/' + info[1] + f'/results/{dataset}/{id}_{suffix}.txt'
    with open(fn, 'r') as fd:
        pred_lines = fd.readlines()
    pred_lines = [
        x.strip() for x in pred_lines if len(x.strip()) > 0
    ]
    # TODO investigate
    return pred_lines


def get_llama_fns(info, dataset):
    suffix = info[-1]
    fns = list(glob('out/adapter_v2/' + info[1] + f'/results/{dataset}/*{suffix}.txt'))
    ids = [
        fn.split('/')[-1].replace('.txt', '').split('_')[0] for fn in fns
    ]
    return list(zip(ids, fns))
