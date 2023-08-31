from collections import defaultdict
from datasets import load_dataset, load_from_disk
from evaluate import load
from nltk import word_tokenize, sent_tokenize

import spacy
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
from scipy.stats import pearsonr
from time import sleep
from nltk.util import ngrams
from rouge_utils import *
from fragments import compute_frags

from data_utils import *
from oa_secrets import OA_KEY, OA_ORGANIZATION
import openai

openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY
import backoff


PREFIX = 'Here is an Article along with several possible summaries for the article.'
SUFFIXES = {
    'informative': 'Please rate the summary (1=worst to 5=best) with respect to informativeness. An informative summary captures the important information in the article and presents it accurately and concisely. Return a single number.',
    'quality': 'Please rate the summary (1=worst to 5=best) with respect to quality. A high quality summary is comprehensible and understandable. Return a single number.',
    'attributable': 'Please rate the summary (1=worst to 5=best) with respect to attribution. Is all the information in the summary fully attributable to the Article? Return a single number.',
    'coherence': 'Please rate the summary (1=worst to 5=best) with respect to coherence. A coherent summary is well-structured and well-organized. Return a single number.',
    'overall': 'Please rate the summary (1=worst to 5=best) with respect to overall preference. A good summary should convey the main ideas in the Article in a concise, logical, and coherent fashion. Return a single number.',
}


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError), max_tries=25)
def chatgpt(messages, model='gpt-4', max_tokens=32):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']


def num_ents(text, nlp):
    return len(list(set(list(nlp(text).ents))))


def redundancy(text):
    toks = word_tokenize(text)
    unigrams = list(ngrams(toks, 1))
    bigrams = list(ngrams(toks, 2))
    trigrams = list(ngrams(toks, 3))

    return {
        'unique_unigrams': len(set(unigrams)) / len(unigrams),
        'unique_bigrams': len(set(bigrams)) / len(bigrams),
        'unique_trigrams': len(set(trigrams)) / len(trigrams),
    }


def compute_exp(nlp, rouge, name, sources, source_tokens, references, preds):
    print(f'Starting {name}')
    exp_stats = defaultdict(list)
    tokens = [
        len(word_tokenize(pred)) for pred in preds
    ]
    exp_stats['tokens'] = tokens
    if len(tokens) == len(source_tokens):
        exp_stats['length_correlation'] = pearsonr(tokens, source_tokens)[0]

    for p, r in zip(preds, references):
        obj = rouge.compute(predictions=[p], references=[r], use_aggregator=False)
        for k, v in obj.items():
            exp_stats[k].append(v[0])

    for dimension, prompt in SUFFIXES.items():
        print(f'Starting GPT-4 {dimension} for {name}')
        scores = []
        for source, pred in zip(sources, preds):
            if dimension in {'quality', 'coherence'}:
                prompt = f'{PREFIX}\n\nSummary: {pred}\n\n{SUFFIXES[dimension]}'
            else:
                prompt = f'{PREFIX}\n\nArticle: {source}\n\nSummary: {pred}\n\n{SUFFIXES[dimension]}'

            messages = [
                # Boost its ego first
                {'role': 'system', 'content': 'You are an evaluator of text summaries.'},
                {'role': 'user', 'content': prompt}
            ]

            scores.append(float(chatgpt(messages=messages, model='gpt-4').strip()))
            sleep(4)

        exp_stats[f'gpt4_{dimension}_grade'] = scores

    # Create the histogram
    sns.histplot(tokens, bins=20, kde=True)

    plt.title(f'Distribution of tokens for {name}')

    # Save the plot to 'save.png'
    # plt.savefig(f"{name}_tokens_hist.png")

    # plt.clf()

    frags = [compute_frags({'source': source, 'prediction': pred}) for source, pred in zip(sources, preds)]

    exp_stats['coverage'] = [f['coverage'] for f in frags]
    exp_stats['density'] = [f['density'] for f in frags]

    out = [redundancy(pred) for pred in preds]
    exp_stats['num_ents'] = [num_ents(pred, nlp) for pred in preds]
    exp_stats['num_ents_per_token'] = [a / b for a, b in zip(exp_stats['num_ents'], exp_stats['tokens'])]
    exp_stats['unique_unigrams'] = [x['unique_unigrams'] for x in out]
    exp_stats['unique_bigrams'] = [x['unique_bigrams'] for x in out]
    exp_stats['unique_trigrams'] = [x['unique_trigrams'] for x in out]

    all_pred_sents = [sent_tokenize(pred) for pred in preds]
    for source, pred_sents in zip(sources, all_pred_sents):
        source_sents = sent_tokenize(source)
        source_sents_no_stop = [
            remove_stopwords(x) for x in source_sents
        ]
        aligned_idxs = []
        for sent in pred_sents:
            source_idxs, _ = gain_rouge(sent, source_sents_no_stop, max_steps=5)
            aligned_idxs.append(source_idxs)
        fusion_score = np.mean([len(x) for x in aligned_idxs])
        avg_rank = float(np.mean(list(chain(*aligned_idxs))))

        exp_stats['fusion'].append(fusion_score)
        exp_stats['avg_sent_rank'].append(avg_rank)
    print(name)
    row_str = []
    keys = []
    for k, v in exp_stats.items():
        keys.append(k)
        print(k, np.mean(v))
        row_str.append(str(np.mean(v)))
    print('\n\n' + '***START***' + '\n')
    print(name)
    print(' '.join(keys))
    print(' '.join(row_str))
    print('\n' + '***END***' + '\n\n')

    return exp_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--max_examples', default=100, type=int)
    parser.add_argument('--out_fn', default='llama_metrics.json')

    args = parser.parse_args()

    args.out_fn = os.path.expanduser(args.out_fn)
    overwrite = args.overwrite

    rouge = load('rouge', keep_in_memory=True)

    nlp = spacy.load("en_core_web_sm")

    EXPERIMENTS = LLAMA_EXPERIMENTS
    get_fns = get_llama_fns
    get_preds = get_llama_preds
    split = 'test'

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

    sources = [id2article[id] for id in shared_ids]
    source_tokens = [len(word_tokenize(source)) for source in sources]

    out = {'id': shared_ids}

    # compute_exp(nlp, rouge, 'reference', sources, source_tokens, references, references)

    id2preds = defaultdict(dict)

    for exp in EXPERIMENTS:
        preds = [get_preds(exp, id) for id in shared_ids]
        name = exp[0]
        out[name] = compute_exp(nlp, rouge, name, sources, source_tokens, references, preds)

        for id, pred in zip(shared_ids, preds):
            id2preds[id][exp[0]] = pred

    with open(args.out_fn, 'w') as fd:
        json.dump(out, fd)

    outputs = []
    metas = []
    rand_outputs = []
    for j, id in enumerate(shared_ids):
        row = f'ID: {id}\n\n'
        article = id2article[id]
        row += f'Article:\n{article}\n\n'
        for name in [x[0] for x in EXPERIMENTS]:
            row += f'Summary {name}: {id2preds[id][name]}\n\n'

        row += 'Preference:\n'
        row += 'Reason:\n\n'
        outputs.append(row)

        order = np.arange(3)
        np.random.shuffle(order)

        ordered_names = [EXPERIMENTS[rand_idx][0] for rand_idx in order]

        dense_random = [id2preds[id][name] for name in ordered_names]
        meta = {'idx': j, 'id': id, 'order': ','.join(ordered_names)}
        for rank, name in enumerate(ordered_names):
            meta[f'Summary {rank + 1}'] = name
        metas.append(meta)
        rand_row = f'ID: {id}\n\n'
        rand_row += f'Article:\n{article}\n\n'
        for i, d in enumerate(dense_random):
            rand_row += f'Summary {i + 1}: {d}\n\n'
        rand_row += 'Preference:\n'
        rand_row += 'Reason:\n\n'
        rand_outputs.append(rand_row)

    delim = '*' * 75 + '\n\n'
    with open('llama_oracle.txt', 'w') as fd:
        fd.write(delim.join(outputs))

    with open('llama_human.txt', 'w') as fd:
        fd.write(delim.join(rand_outputs))
