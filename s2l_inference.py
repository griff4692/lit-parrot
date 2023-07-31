import argparse
import os

import json
import sys
import time
from datasets import load_dataset
from functools import partial
from pathlib import Path
from typing import Literal
import pandas as pd
from tqdm import tqdm
import numpy as np

import lightning as L
from evaluate import load
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.adapter import Block
from lit_gpt.adapter import GPT, Config
from lit_gpt.adapter_v2 import add_adapter_v2_parameters_to_linear_layers
from lit_gpt.utils import lazy_load, quantization

import glob


ALPACA_HEADER = 'Below is an instruction that describes a task, paired with an input that provides further context. ' \
                'Write a response that appropriately completes the request.'

INSTRUCTIONS = {
    'vanilla': 'Generate a concise and informative Summary of the Article.',
    'tldr': 'Generate a 1 sentence Summary of the Article.',
    'decide_length': 'Is this Summary too short? Answer Y/N.',
    'lengthen': 'Lengthen this Summary with important, non-redundant concepts in the Article.'
}


def get_latest_file(directory):
    # Get list of all files, and sort them by modified time descending
    files = sorted(glob.glob(os.path.join(directory, '*.pth')), key=os.path.getmtime, reverse=True)
    # If list is not empty, return the first file (which will have the latest timestamp)
    if files:
        print(f'Using ', files[0])
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
    parser.add_argument('--adapter_path', default=None)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--max_new_tokens', default=368, type=int)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--precision', default='bf16-true')
    parser.add_argument('--max_examples', default=100, type=int)
    parser.add_argument('-overwrite', default=False, action='store_true')

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

    torch.set_float32_matmul_precision("high")

    args.checkpoint_dir = Path(args.checkpoint_dir)

    if args.adapter_path is None:
        adapter_path = None
        results_dir = os.path.join('out', args.base)
    else:
        results_dir = os.path.join(args.adapter_path, 'results')
        args.adapter_path = Path(args.adapter_path)
        adapter_path = get_latest_file(args.adapter_path)
    os.makedirs(results_dir, exist_ok=True)

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

    if adapter_path is None:
        t0 = time.time()
        with lazy_load(checkpoint_path) as checkpoint:
            model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        fabric.print(f"Time to load the model weights: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    else:
        t0 = time.time()
        with lazy_load(checkpoint_path) as checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
            checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
            model.load_state_dict(checkpoint, strict=True)
        fabric.print(f"Time to load the model weights: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)
    tokenizer = Tokenizer(args.checkpoint_dir)

    print('Reading in dataset...')
    dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
    n = len(dataset)
    if n > args.max_examples:
        np.random.seed(1992)
        sample = list(sorted(np.random.choice(np.arange(n), size=(args.max_examples,), replace=False)))
        dataset = dataset.select(sample)

    for example in tqdm(dataset, total=len(dataset)):
        progressive_predictions = []
        id = example['id']
        article = example['article']
        article_toks = article.split(' ')
        n = len(article_toks)
        if n > args.max_article_toks:
            article = ' '.join(article_toks[:args.max_article_toks])

        tldr_input = f'Article: {article}'
        tldr_prompt = f"{ALPACA_HEADER}\n\n### Instruction:\n{INSTRUCTIONS['tldr']}\n\n### Input:\n{tldr_input}\n\n### Response:\n"

        out_fn = os.path.join(results_dir, f'{id}_s2l.txt')
        if os.path.exists(out_fn):
            if args.overwrite:
                print(f'Overwriting {out_fn}')
            else:
                print(f'Skipping {out_fn}...')
                continue

        try:
            progressive_predictions = [get_completion(args, model, tokenizer, tldr_prompt)]
            cot = set()
            for iter in range(3):
                change_input = f'Article: {article}\n\nSummary: {progressive_predictions[-1]}'
                decide_prompt = f"{ALPACA_HEADER}\n\n### Instruction:\n{INSTRUCTIONS['decide_length']}\n\n### Input:\n{change_input}\n\n### Response:\n"

                decision = get_completion(args, model, tokenizer, decide_prompt).lower()
                assert len(decision) == 1
                if decision == 'n':
                    break

                update_prompt = f"{ALPACA_HEADER}\n\n### Instruction:\n{INSTRUCTIONS['lengthen']}\n\n### Input:\n{change_input}\n\n### Response:\n"
                prediction = get_completion(args, model, tokenizer, update_prompt)
                pred_lines = prediction.split('\n')
                new_cot = [x.strip() for x in pred_lines[0].split(';')]
                new_cot_valid = [x for x in new_cot if x not in cot]

                if len(new_cot_valid) == 0:
                    print('Predicted same COT. Breaking')
                    break
                elif len(new_cot_valid) < len(new_cot):
                    print('Removing repeated entries...')
                    update_prompt_trunc = update_prompt + 'MISSING: ' + '; '.join(new_cot_valid) + '\nSUMMARY V2: '
                    prediction = get_completion(args, model, tokenizer, update_prompt_trunc)
                    pred_lines = prediction.split('\n')

                for x in new_cot_valid:
                    cot.add(x)
                progressive_predictions.append(pred_lines[0])
                assert len(pred_lines) > 1
                prediction = pred_lines[1].replace('SUMMARY V2:', '').strip()
                progressive_predictions.append(prediction)
            fabric.print(progressive_predictions[-1])
            with open(out_fn, 'w') as fd:
                fd.write('\n'.join(progressive_predictions))
        except Exception as msg:
            print('Failed to parse output...')
            print(msg)
