import argparse
import os

import json
import sys
import time
from datasets import load_dataset, load_from_disk
from functools import partial
from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np

import lightning as L
from evaluate import load
import torch
from tqdm import tqdm
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate, remove_trailing_sent_frag
from lit_gpt import Tokenizer
from lit_gpt.adapter import Block
from lit_gpt.adapter import GPT, Config
from lit_gpt.adapter_v2 import add_adapter_v2_parameters_to_linear_layers
from lit_gpt.utils import lazy_load, quantization

import glob


ALPACA_HEADER = 'Below is an instruction that describes a task, paired with an input that provides further context. ' \
                'Write a response that appropriately completes the request.'

INSTRUCTIONS = {
    # 'rank': 'Rank Summaries of an Article from best to worst.',
    # 'cot': 'Generate a progressively longer summary of the Article.',
    # 'control_length': 'Generate a summary of the Article with a target length.',
    'vanilla': 'Generate a concise and informative summary of the Article.',
    'tldr': 'Generate a one sentence summary of the Article.',
    'change_length': 'Shorten or lengthen a Summary of an Article.'
}


def get_latest_file(directory):
    # Get list of all files, and sort them by modified time descending
    files = sorted(glob.glob(os.path.join(directory, '*.pth')), key=os.path.getmtime, reverse=True)
    # If list is not empty, return the first file (which will have the latest timestamp)
    if files:
        return files[0]
    else:
        return None


def get_completion(args, model, tokenizer, prompt, max_new_tokens=None):
    encoded = tokenizer.encode(prompt, device=model.device)
    prompt_length = encoded.size(0)
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_new_tokens

    if args.base == 'falcon':
        max_seq_length = 2048
        max_returned_tokens = min(2048, prompt_length + max_new_tokens)
    else:
        max_seq_length = 4096
        max_returned_tokens = min(4096, prompt_length + max_new_tokens)

    y = generate(
        model,
        encoded,
        max_returned_tokens,
        max_seq_length=max_seq_length,
        temperature=args.temperature,
        eos_id=tokenizer.eos_id,
    )
    model.reset_cache()
    output = tokenizer.decode(y)
    _, prediction = output.split('### Response:')
    return prediction.strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--base', default='llama')
    parser.add_argument('--adapter_path', default='out/adapter_v2/s2l_llama_gpt4_selection')
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('--max_new_tokens', default=160, type=int)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--precision', default='bf16-true')
    parser.add_argument('--max_examples', default=100, type=int)
    parser.add_argument('--split', default='test')
    parser.add_argument('--return_candidates', default=1, type=int)

    args = parser.parse_args()

    if args.base == 'llama':
        args.checkpoint_dir = 'checkpoints/meta-llama/Llama-2-7b-hf'
        args.max_article_toks = 2048
    elif args.base == 'llama_chat':
        args.checkpoint_dir = 'checkpoints/meta-llama/Llama-2-7b-chat-hf'
        args.max_article_toks = 2048
    else:
        args.checkpoint_dir = 'checkpoints/tiiuae/falcon-7b'
        args.max_article_toks = 1024

    print(f'Inferring checkpoint dir of {args.checkpoint_dir}')

    args.checkpoint_dir = Path(args.checkpoint_dir)
    torch.set_float32_matmul_precision("high")

    if args.adapter_path is None:
        adapter_path = None
        results_dir = os.path.join('out', args.base, args.dataset)
    else:
        results_dir = os.path.join(args.adapter_path, 'results', args.dataset)
        args.adapter_path = Path(args.adapter_path)
        adapter_path = get_latest_file(args.adapter_path)

    os.makedirs(results_dir, exist_ok=True)
    if args.split == 'train':
        os.makedirs(os.path.join(results_dir, 'train'), exist_ok=True)
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, cpu_offload=False)
    fabric = L.Fabric(devices=args.devices, precision=args.precision, strategy=strategy)
    fabric.launch()

    with open(args.checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    model_file = "lit_model.pth"
    checkpoint_path = args.checkpoint_dir / model_file

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)

    t0 = time.time()
    with fabric.init_module(empty_init=True), quantization(None):
        model = GPT(config)
        if adapter_path is not None:
            add_adapter_v2_parameters_to_linear_layers(model)
    fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.time()
    if adapter_path is None:
        with lazy_load(checkpoint_path) as checkpoint:
            model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    else:
        with lazy_load(checkpoint_path) as checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
            checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
            model.load_state_dict(checkpoint, strict=True)
    fabric.print(f"Time to load the model weights: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(args.checkpoint_dir)

    print(args.max_new_tokens)
    print('Reading in dataset...')
    if args.dataset == 'cnn':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split=args.split)
    elif args.dataset == 'xsum':
        dataset = load_dataset(args.dataset, split=args.split)
    else:
        dataset = load_from_disk(os.path.expanduser('~/nyt_edu_alignments'))[args.split]
        dataset = dataset.rename_columns({
            'article_untok': 'document',
            'abstract_untok': 'summary'
        })

    n = len(dataset)
    if n > args.max_examples:
        np.random.seed(1992)
        sample = list(sorted(np.random.choice(np.arange(n), size=(args.max_examples,), replace=False)))
        dataset = dataset.select(sample)

    for example in tqdm(dataset, total=len(dataset)):
        id = example['id']
        article = example['article'] if 'article' in example else example['document']
        article_toks = article.split(' ')
        n = len(article_toks)
        if n > args.max_article_toks:
            article = ' '.join(article_toks[:args.max_article_toks])

        summarize_input = f'Article: {article}'
        summarze_prompt = f"{ALPACA_HEADER}\n\n### Instruction:\n{INSTRUCTIONS['vanilla']}\n\n### Input:\n{summarize_input}\n\n### Response:\n"
        predictions = []

        try:
            for _ in range(args.return_candidates):
                prediction = remove_trailing_sent_frag(get_completion(
                    args, model, tokenizer, summarze_prompt, max_new_tokens=args.max_new_tokens
                ))
                predictions.append(prediction)
        except Exception as e:
            print(e)
            continue

        fabric.print('\n'.join(predictions))

        if args.split == 'train':
            out_fn = os.path.join(results_dir, 'train', f'{id}.txt')
        else:
            out_fn = os.path.join(results_dir, f'{id}_summarize.txt')
        with open(out_fn, 'w') as fd:
            fd.write('\n'.join(predictions))
