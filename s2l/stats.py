from collections import defaultdict
from datasets import load_dataset, load_from_disk
from evaluate import load
from nltk import word_tokenize, sent_tokenize

import spacy
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
from scipy.stats import pearsonr
from nltk.util import ngrams
from rouge_utils import *
from fragments import compute_frags

from data_utils import *


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


def compute_exp(nlp, name, sources, source_tokens, preds):
    exp_stats = defaultdict(list)
    tokens = [
        len(word_tokenize(pred)) for pred in preds
    ]
    exp_stats['tokens'] = tokens
    if len(tokens) == len(source_tokens):
        exp_stats['length_correlation'] = pearsonr(tokens, source_tokens)[0]

    # Create the histogram
    sns.histplot(tokens, bins=20, kde=True)

    plt.title(f'Distribution of tokens for {name}')

    # Save the plot to 'save.png'
    plt.savefig(f"{name}_tokens_hist.png")

    plt.clf()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnn')
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--max_examples', default=100, type=int)
    parser.add_argument('--model_class', default='gpt4', choices=['llama', 'gpt4'])

    args = parser.parse_args()

    overwrite = args.overwrite

    rouge = load('rouge', keep_in_memory=True)

    nlp = spacy.load("en_core_web_sm")

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

    stats_dir = os.path.expanduser(f'~/lit-parrot/out/eval/stats')
    os.makedirs(stats_dir, exist_ok=True)

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
    for exp in EXPERIMENTS:
        preds = [get_preds(exp, id) for id in shared_ids]
        num_preds_per = len(preds[0])
        for step in range(num_preds_per):
            step_preds = [pred[step] for pred in preds if step < len(pred)]
            if num_preds_per == 1:
                name = exp[0]
            else:
                name = exp[0] + f'_step_{step + 1}'
            compute_exp(nlp, name, sources, source_tokens, step_preds)

    compute_exp(nlp, 'reference', sources, source_tokens, references)
