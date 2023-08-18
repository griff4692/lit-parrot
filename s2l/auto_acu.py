import argparse
from datasets import load_dataset, load_from_disk
import numpy as np
from autoacu import A3CU, A2CU
import torch


from data_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--experiment', default='default')
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--max_examples', default=100, type=int)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--model_class', default='gpt4', choices=['llama', 'gpt4'])

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'

    args.experiment += '_' + args.dataset

    overwrite = args.overwrite

    a3cu = A3CU(device=args.device)  # the GPU device to use
    # a2cu = A2CU(device=args.device)

    EXPERIMENTS = GPT4_EXPERIMENTS if args.model_class == 'gpt4' else LLAMA_EXPERIMENTS
    get_fns = get_gpt4_fns if args.model_class == 'gpt4' else get_llama_fns
    get_preds = get_gpt4_preds if args.model_class == 'gpt4' else get_llama_preds
    split = 'train' if args.model_class == 'gpt4' else 'test'

    experiment_fns = [
        get_fns(info) for info in EXPERIMENTS
    ]

    print([
        (EXPERIMENTS[i][0], len(experiment_fns[i])) for i in range(len(experiment_fns))
    ])

    shared_ids = set([x[0] for x in experiment_fns[0]])
    for i in range(2, len(experiment_fns)):
        shared_ids.intersection(set([x[0] for x in experiment_fns[i]]))

    shared_ids = list(sorted(list(shared_ids)))
    assert len(shared_ids) >= args.max_examples

    if len(shared_ids) > args.max_examples:
        np.random.seed(1992)
        np.random.shuffle(shared_ids)
        shared_ids = shared_ids[:args.max_examples]

    acu_dir = os.path.expanduser(f'~/lit-parrot/out/eval/{args.experiment}/acu')
    os.makedirs(acu_dir, exist_ok=True)

    print('Reading in dataset...')
    if args.dataset == 'cnn':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split=split)
    elif args.dataset == 'xsum':
        dataset = load_dataset(args.dataset, split=split)
    else:
        dataset = load_from_disk(os.path.expanduser('~/nyt_edu_alignments'))[split]
        dataset = dataset.rename_columns({
            'article_untok': 'document',
            'abstract_untok': 'summary'
        })

    id2article = dict(zip(dataset['id'], dataset['article'] if 'article' in dataset.features else dataset['document']))
    id2reference = dict(
        zip(dataset['id'], dataset['highlights'] if 'highlights' in dataset.features else dataset['summary']))

    references = [
        id2reference[id] for id in shared_ids
    ]

    for exp in EXPERIMENTS:
        preds = [
            get_preds(exp, id) for id in shared_ids
        ]
        num_preds_per = len(preds[0])
        for step in range(num_preds_per):
            step_preds = [pred[step] for pred in preds]
            if num_preds_per == 1:
                name = exp[0]
            else:
                name = exp[0] + f'_step_{step + 1}'
            recall_scores, prec_scores, f1_scores = a3cu.score(
                references=references,
                candidates=step_preds,
                batch_size=16,  # the batch size for ACU generation
                output_path=None  # the path to save the evaluation results
            )
            print(name)
            print(f"Recall: {np.mean(recall_scores):.4f}, Precision {np.mean(prec_scores):.4f}, F1: {np.mean(f1_scores):.4f}")
            print('\n\n\n\n\n\n')
