import os
import json
from glob import glob


LLAMA_EXPERIMENTS = [
    ['dense_llama_chat', 'dense_llama_chat', 'cnn_final'],
    ['baseline_llama_chat', 'baseline_llama_chat', 'cnn'],
    ['zeroshot_llama_chat', 'zeroshot_llama_chat', 'cnn'],
]


GPT4_EXPERIMENTS = [
    ['human_dense_v3'],
    # ['dense_train_v2'],
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


def get_gpt4_fns(info):
    suffix = info[-1]
    pattern = os.path.join(GPT4_DIR, suffix, '*.json')
    fns = list(glob(pattern))
    ids = [
        fn.split('/')[-1].replace('.json', '').split('_')[0] for fn in fns
    ]
    return list(zip(ids, fns))


def get_llama_preds(info, id):
    is_zero = 'zero' in info[0]
    if is_zero:
        fn = os.path.expanduser('~/lit-parrot/out/adapter_v2/' + info[1] + f'/{info[2]}/{id}_summarize.txt')
    else:
        fn = os.path.expanduser('~/lit-parrot/out/adapter_v2/' + info[1] + f'/results/{info[2]}/{id}_summarize.txt')
    with open(fn, 'r') as fd:
        return fd.read().strip()


def get_llama_fns(info):
    is_zero = 'zero' in info[0]
    if is_zero:
        fns = list(glob(os.path.expanduser('~/lit-parrot/out/adapter_v2/' + info[1] + f'/{info[2]}/*.txt')))
    else:
        fns = list(glob(os.path.expanduser('~/lit-parrot/out/adapter_v2/' + info[1] + f'/results/{info[2]}/*.txt')))
    ids = [
        fn.split('/')[-1].replace('.txt', '').split('_')[0] for fn in fns
    ]
    return list(zip(ids, fns))
