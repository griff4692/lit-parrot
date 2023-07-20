import argparse
import os

import json
import sys
import time
from datasets import load_from_disk
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


def dedup(text):
    sents = text.split('\n')
    seen = set()
    new = []
    for sent in sents:
        if sent.lower() in seen:
            continue
        seen.add(sent.lower())
        new.append(sent)
    return '\n'.join(new)


def parse(output):
    output = output.split('### Length:')[1].strip()

    if len(output) < 3:
        try:
            length = int(output)
        except:
            length = None
        extract = ''
        abstract = ''
    else:
        try:
            length, output = output.split("### Extract:")
            length = int(length.strip())
        except:
            print('Could not parse into length and extract ->', output)
            length = None
        output = output.strip()

        if '### Abstract:' not in output:
            extract = abstract = None
            print('Poorly formed output. Setting to None.')
        else:
            try:
                extract, abstract = output.split("### Abstract:")
                extract = extract.strip()
                extract = dedup(extract)
                abstract = abstract.strip()
                abstract = dedup(abstract)
            except:
                print('Too many values to unpack in ', output)
                extract = abstract = None
    return {'length': length, 'extract': extract, 'abstract': abstract}


def generate_notewise(
        model, tokenizer, notes, max_draft_tokens=256, max_update_tokens=256, max_note_toks=1024, temperature=0.8
):
    for i in range(len(notes)):
        num_toks = len(notes[i].split(' '))
        if num_toks > max_note_toks:
            notes[i] = ' '.join(notes[i].split(' ')[:max_note_toks])

    draft = ''
    for note in notes:
        prompt = f'### Note:\n{note}\n\n### Draft:\n{draft}\n\n### Length:\n'

        encoded = tokenizer.encode(prompt, device=model.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = min(3072, prompt_length + max_update_tokens)

        y = generate(
            model,
            encoded,
            max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=temperature,
            eos_id=tokenizer.eos_id,
        )
        model.reset_cache()
        output = tokenizer.decode(y)
        parsed = parse(output)

        if parsed['abstract'] is not None and len(parsed['abstract'].split(' ')) >= 3:
            draft_sents = [x.strip() for x in draft.split('\n') if len(x.strip()) > 0]
            new_sents = parsed['abstract'].split('\n')
            for new_sent in new_sents:
                if new_sent not in draft_sents:
                    draft_sents.append(new_sent)

            draft = '\n'.join(draft_sents).strip()

        draft_toks = len(draft.split(' '))
        if draft_toks >= max_draft_tokens:
            print(f'Breaking: {draft_toks}/{max_tokens} tokens')
            break

    return draft


def batch_generate_notewise(model, tokenizer, dataset, max_notes_to_update=5):
    rouge = load('rouge', keep_in_memory=True)
    outputs = []
    for example in dataset:
        source = example['source']
        notes = source.split('<doc-sep>')
        notes = [x.replace('<s>', '').replace('</s>', '').strip() for x in notes[1:]]
        if len(notes) > max_notes_to_update:
            print(f'Using first {max_notes_to_update}/{len(notes)} notes...')
            notes = notes[:max_notes_to_update]
        prediction = generate_notewise(model, tokenizer, notes)
        example['prediction'] = prediction

        if prediction is None:
            example['rouge1'] = example['rouge2'] = None
            print('Could not parse prediction.')
        else:
            reference = '\n'.join(example['target_sents'])
            robj = rouge.compute(references=[reference], predictions=[prediction], use_aggregator=False)
            example['rouge1'] = robj['rouge1'][0]
            example['rouge2'] = robj['rouge2'][0]

        outputs.append(example)
    outputs = pd.DataFrame(outputs)
    print('ROUGE 1: ', outputs['rouge1'].dropna().mean())
    print('ROUGE 2: ', outputs['rouge2'].dropna().mean())
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--adapter_path', default=Path('out/adapter_v2/bhc_alpaca_update/iter-019199-ckpt.pth'))
    parser.add_argument('--checkpoint_dir', default=Path('checkpoints/tiiuae/falcon-7b'))
    parser.add_argument('--strategy', default='fsdp')
    parser.add_argument('--data_dir', default='data/bhc_update')
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--max_new_tokens', default=256, type=int)
    parser.add_argument('--precision', default='bf16-true')

    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    if args.strategy == "fsdp":
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, cpu_offload=False)
    else:
        raise Exception('Nope')
    fabric = L.Fabric(devices=args.devices, precision=args.precision, strategy=strategy)
    fabric.launch()

    with open(args.checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    config.block_size = 3072

    model_file = "lit_model.pth"
    checkpoint_path = args.checkpoint_dir / model_file

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.time()
    with fabric.init_module(empty_init=True), quantization(None):
        model = GPT(config)
        add_adapter_v2_parameters_to_linear_layers(model)
    fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.time()
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(args.adapter_path) as adapter_checkpoint:
        checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
        model.load_state_dict(checkpoint, strict=True)
    fabric.print(f"Time to load the model weights: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(args.checkpoint_dir / "tokenizer.json", args.checkpoint_dir / "tokenizer_config.json")

    print('Reading in dataset...')
    debug = True
    mini_str = '_mini' if debug else ''
    data_dir = os.path.expanduser(f'~/partials{mini_str}')
    print(f'Reading in data from {data_dir}')
    dataset = load_from_disk(data_dir)['test']

    outputs = batch_generate_notewise(model, tokenizer, dataset)

    print(outputs['prediction'].tolist())
