"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json
import os.path
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from p_tqdm import p_uimap

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer
from datasets import load_from_disk, concatenate_datasets


ORACLE_STRATEGY = 'update'  # gain
IN_DIRS = [
    f'/nlp/projects/summarization/note_partials_oracle_{ORACLE_STRATEGY}_llm',
    f'/nlp/data/cdr/epic_docs_2020_20230625/partial_summarization_dataset_oracle_{ORACLE_STRATEGY}_llm'
]

DESTINATION_PATH = Path(os.path.expanduser(f"/nlp/data/cdr/epic_docs_2020_20230625/llama_data"))
CHECKPOINT_DIR = Path(os.path.expanduser("~/lit-parrot/checkpoints/meta-llama/Llama-2-7b-hf"))
IGNORE_INDEX = -1
MASK_INPUTS = False  # as in alpaca-lora
SEED = 42
MAX_LENGTH = 4096  # useful to know the minimum max_seq_length during fine-tuning (saves memory!)


def prepare(
    destination_path: Path = DESTINATION_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    mask_inputs: bool = MASK_INPUTS,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    destination_path.mkdir(parents=True, exist_ok=True)

    print(f'Loading datasets from {IN_DIRS}')
    train_sets = []
    eval_sets = []
    for dir in IN_DIRS:
        dataset = load_from_disk(dir)
        train = dataset['train']
        eval = dataset['eval']
        train_sets.append(train)
        eval_sets.append(eval)
    train_set = concatenate_datasets(train_sets).shuffle(seed=1992)
    eval_set = concatenate_datasets(eval_sets)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    n_train = len(train_set)
    n_test = len(eval_set)

    print(f"train has {n_train} samples")
    print(f"val has {n_test} samples")

    print("Processing train split ...")
    train_set = list(p_uimap(
        lambda sample:
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        ), train_set
    ))

    train_set = list(filter(None, train_set))
    print(f"train has {len(train_set)}/{n_train} samples after removing too long.")
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    eval_set = list(p_uimap(
        lambda sample:
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        ), eval_set
    ))
    eval_set = list(filter(None, eval_set))
    print(f"train has {len(eval_set)}/{n_test} samples after removing too long.")
    torch.save(eval_set, destination_path / "test.pt")

    with open(destination_path / "config.json", "w") as file:
        json.dump({"max_seq_length": MAX_LENGTH}, file)


def prepare_sample(
    example: dict,
    tokenizer: Tokenizer,
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
    full_prompt = example["input"]
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=MAX_LENGTH)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=MAX_LENGTH)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    if encoded_full_prompt_and_response.shape[-1] == MAX_LENGTH:
        print('Input too long. Return None')
        return None

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


if __name__ == "__main__":
    prepare()
