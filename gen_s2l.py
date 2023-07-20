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


def get_latest_file(directory):
    # Get list of all files, and sort them by modified time descending
    files = sorted(glob.glob(os.path.join(directory, '*.pth')), key=os.path.getmtime, reverse=True)
    # If list is not empty, return the first file (which will have the latest timestamp)
    if files:
        return files[0]
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--adapter_path', default=Path('out/adapter_v2/s2l'))
    parser.add_argument('--checkpoint_dir', default=Path('checkpoints/tiiuae/falcon-7b'))
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--max_new_tokens', default=256, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--precision', default='bf16-true')

    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    results_dir = os.path.join(args.adapter_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    adapter_path = get_latest_file(args.adapter_path)

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
        add_adapter_v2_parameters_to_linear_layers(model)
    fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.time()
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
        model.load_state_dict(checkpoint, strict=True)
    fabric.print(f"Time to load the model weights: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(args.checkpoint_dir / "tokenizer.json", args.checkpoint_dir / "tokenizer_config.json")

    print('Reading in dataset...')
    dataset = load_dataset('griffin/progressive_summarization')['eval']

    for example in dataset:
        id = example['id']
        task = example['task']
        prompt = example['prompt']
        encoded = tokenizer.encode(prompt, device=model.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = min(2048, prompt_length + args.max_new_tokens)

        y = generate(
            model,
            encoded,
            max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=args.temperature,
            eos_id=tokenizer.eos_id,
        )
        model.reset_cache()
        output = tokenizer.decode(y)
        try:
            _, prediction = output.split('### Response:')
        except:
            print(f'Output has too many Response strings. Ignoring for now. -> {output}')
            predictions = output.split('### Response:')
            prediction = '\n'.join(predictions[1:])

        prediction = prediction.strip()
        fabric.print(prediction)

        out_fn = os.path.join(results_dir, f'{id}_{task}.txt')
        with open(out_fn, 'w') as fd:
            fd.write(prediction)
