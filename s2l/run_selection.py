import os
from glob import glob
import json

import backoff

from oa_secrets import OA_KEY, OA_ORGANIZATION
import openai
from tqdm import tqdm

openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError), max_tries=20)
def chatgpt(messages, model='gpt-4', max_tokens=256):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.1, max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    pattern = os.path.expanduser(f'~/Desktop/s2l_data/cnn/s2l_rollout_v2_train/*.json')
    fns = list(glob(pattern))

    prompt = open('selection_prompt.txt').read()

    for fn in tqdm(fns):
        out_fn = fn.replace('.json', '') + '_selection.txt'
        # if os.path.exists(out_fn):
        #     print(f'{out_fn} exists. Skipping...')
        #     continue

        with open(fn, 'r') as fd:
            data = json.load(fd)
            article = data['article']
            summaries = data['prediction']
            missing = data['missing']

            this_prompt = prompt.replace("{{ARTICLE}}", article)
            this_prompt = this_prompt.replace("{{S1}}", summaries[0])

            this_prompt = this_prompt.replace("{{M2}}", missing[1])
            this_prompt = this_prompt.replace("{{S2}}", summaries[1])

            this_prompt = this_prompt.replace("{{M3}}", missing[2])
            this_prompt = this_prompt.replace("{{S3}}", summaries[2])

            this_prompt = this_prompt.replace("{{M4}}", missing[3])
            this_prompt = this_prompt.replace("{{S4}}", summaries[3])

            messages = [
                # Boost its ego first
                {'role': 'system', 'content': 'You are a discerning evaluator of text summaries.'},
                {'role': 'user', 'content': this_prompt}
            ]

            output = chatgpt(messages)
            print(output)

            with open(out_fn, 'w') as fd:
                fd.write(output)