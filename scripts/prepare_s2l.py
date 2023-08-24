"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json
import os.path
import sys
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer
from datasets import load_dataset


MAIN_DIR = os.path.expanduser(f"~/lit-parrot")
PARROT_MODEL = os.environ['PARROT_MODEL']
if PARROT_MODEL == 'falcon':
    CHECKPOINT_DIR = Path(os.path.join(MAIN_DIR, "checkpoints/tiiuae/falcon-7b"))
else:
    CHECKPOINT_DIR = Path(os.path.join(MAIN_DIR, "checkpoints/meta-llama/Llama-2-7b-hf"))
IGNORE_INDEX = -1
MASK_INPUTS = False
SEED = 42
MAX_LENGTH = 0  # useful to know the minimum max_seq_length during fine-tuning (saves memory!)


def prepare(
    args,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    mask_inputs: bool = MASK_INPUTS,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    with open(checkpoint_dir / "lit_config.json", "r") as file:
        config = json.load(file)
        max_seq_length = min(2048, config["block_size"])

    destination_path = Path(os.path.join(MAIN_DIR, f"data/{args.dataset}_{PARROT_MODEL}"))

    destination_path.mkdir(parents=True, exist_ok=True)

    print(f'Loading dataset...')

    if args.dataset == 'dense':
        train_set = load_dataset('griffin/dense_summ', split='train')
    else:
        assert args.dataset == 'length'
        dataset = load_dataset('griffin/length_summarization')
        train_set = dataset['train']

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    n_train = len(train_set)

    print(f"train has {n_train} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]

    train_set = list(filter(None, train_set))
    print(f"train has {len(train_set)}/{n_train} samples after removing too long.")
    torch.save(train_set, destination_path / "train.pt")

    print(destination_path)

    with open(destination_path / "config.json", "w") as file:
        json.dump({"max_seq_length": MAX_LENGTH}, file)


def prepare_sample(
    example: dict,
    tokenizer: Tokenizer,
    max_length: int,
    mask_inputs: bool = MASK_INPUTS,
    ignore_index: int = IGNORE_INDEX,
):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = example["prompt"]
    full_prompt_and_response = full_prompt + example["completion"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    global MAX_LENGTH
    MAX_LENGTH = max(MAX_LENGTH, len(encoded_full_prompt_and_response))

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    if encoded_full_prompt_and_response.shape[-1] == max_length:
        print('Warning. Input too long.')

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Prepare')
    parser.add_argument('--dataset', default='dense', choices=['dense', 'length'])
    args = parser.parse_args()
    prepare(args)
