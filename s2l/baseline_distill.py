from datasets import Dataset, DatasetDict
import os
from glob import glob
import argparse
from oa_secrets import HF_ACCESS_TOKEN

import ujson
from tqdm import tqdm


ALPACA_HEADER = 'Below is an instruction that describes a task, paired with an input that provides further context. ' \
                'Write a response that appropriately completes the request.'

INSTRUCTIONS = {
    'summarize': 'Generate an entity-dense summary of the Article.',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--experiment', default='baseline_train')
    args = parser.parse_args()

    import os
    in_dir = os.path.expanduser(f'~/Desktop/s2l_data/cnn/{args.experiment}')

    outputs = []
    fns = list(glob(os.path.join(in_dir, '*.json')))
    for fn in tqdm(fns):
        with open(fn, 'r') as fd:
            result = ujson.load(fd)
            article = result['article'].strip()
            predictions = result['prediction']
            n = len(predictions)
            prediction = predictions[0] if type(predictions) == list else predictions

            input = f'Article: {article}'
            prompt = f"{ALPACA_HEADER}\n\n### Instruction:\n{INSTRUCTIONS['summarize']}\n\n### Input:\n{input}\n\n### Response:\n"
            outputs.append({
                'id': result['id'],
                'prompt': prompt,
                'completion': prediction,
            })

    print('Building dataset from list...')
    splits = DatasetDict({
        'train': Dataset.from_list(outputs)
    })
    print(f"Pushing {len(splits['train'])} train examples to the Hub...")
    splits.push_to_hub('griffin/baseline_summarization', token=HF_ACCESS_TOKEN)
