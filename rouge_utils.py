import os
import string

import argparse
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm
import unicodedata


def decode_utf8(str):
    """
    :param str: string with utf-8 characters
    :return: string with all ascii characters

    This is necessary for the ROUGE nlp package consumption.
    """
    return unicodedata.normalize(u'NFKD', str).encode('ascii', 'ignore').decode('utf8').strip()


STOPWORDS = set(stopwords.words('english')).union(string.punctuation)


def max_rouge_sent(
        target, source_sents, rouge_types, return_score=False, source_prefix='', mask_idxs=[], metric='f1'
):
    n = len(source_sents)
    predictions = [source_prefix + s for s in source_sents]
    references = [target for _ in range(n)]
    outputs = compute(
        predictions=predictions, references=references, rouge_types=rouge_types, use_aggregator=False)
    if metric == 'f1':
        scores = np.array(
            [sum([outputs[t][i].fmeasure for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    elif metric == 'recall':
        scores = np.array(
            [sum([outputs[t][i].recall for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    elif metric == 'precision':
        scores = np.array(
            [sum([outputs[t][i].precision for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    if len(mask_idxs) > 0:
        scores[mask_idxs] = float('-inf')
    max_idx = np.argmax(scores)
    max_source_sent = source_sents[max_idx]
    if return_score:
        return max_source_sent, max_idx, scores[max_idx]
    return max_source_sent, max_idx


def gain_rouge(sent, source_sents_no_stop, max_steps=5):
    # https://aclanthology.org/P19-1209.pdf
    pred_remaining_toks = list(filter(lambda x: len(x) > 0, sent.split(' ')))
    used_idxs = []
    rouge_types = ['rouge1', 'rouge2']
    scores = []
    for extraction_step in range(max_steps):
        _, idx, score = max_rouge_sent(
            ' '.join(pred_remaining_toks), source_sents_no_stop, rouge_types, return_score=True, source_prefix='',
            mask_idxs=used_idxs, metric='f1'
        )
        overlapping_toks = set(pred_remaining_toks).intersection(set(source_sents_no_stop[idx].split(' ')))
        if extraction_step > 0 and len(overlapping_toks) < 2:
            break
        used_idxs.append(idx)
        scores.append(score)
        pred_remaining_toks = [x for x in pred_remaining_toks if x not in overlapping_toks]
        if len(pred_remaining_toks) == 0:
            break
    return used_idxs, scores


def compute(predictions, references, rouge_types=None, use_aggregator=True, use_parallel=False, show_progress=False):
    if rouge_types is None:
        rouge_types = ['rouge1']

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False)
    aggregator = rouge_scorer.scoring.BootstrapAggregator() if use_aggregator else None
    if not use_aggregator and use_parallel:
        scores = list(p_imap(lambda x: scorer.score(x[0], x[1]), list(zip(references, predictions))))
    else:
        scores = []
        if show_progress:
            for i in tqdm(range(len(references))):
                score = scorer.score(references[i], predictions[i])
                if use_aggregator:
                    aggregator.add_scores(score)
                else:
                    scores.append(score)
        else:
            for i in range(len(references)):
                score = scorer.score(references[i], predictions[i])
                if use_aggregator:
                    aggregator.add_scores(score)
                else:
                    scores.append(score)

    if use_aggregator:
        result = aggregator.aggregate()
    else:
        result = {}
        for key in scores[0]:
            result[key] = list(score[key] for score in scores)

    return result


def prepare_str_for_rouge(str):
    return decode_utf8(remove_stopwords(str))


def remove_stopwords(str):
    tok = str.split(' ')
    return ' '.join([t for t in tok if not t in STOPWORDS])


def remove_stopword_toks(toks):
    return list(filter(lambda x: x not in STOPWORDS, toks))


def top_rouge_sents(target, source_sents, rouge_types, metric='f1'):
    def get_metric(obj):
        if metric == 'f1':
            return obj.fmeasure
        elif metric == 'recall':
            return obj.recall
        elif metric == 'precision':
            return obj.precision
        else:
            raise Exception(f'Unknown metric -> {metric}')

    n = len(source_sents)
    target_no_stop = prepare_str_for_rouge(target)
    source_sents_no_stop = list(map(prepare_str_for_rouge, source_sents))
    references = [target_no_stop for _ in range(n)]
    outputs = compute(
        predictions=source_sents_no_stop, references=references, rouge_types=rouge_types, use_aggregator=False)
    scores = np.array([sum([get_metric(outputs[t][i]) for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])

    seen = set()
    for idx, sent in enumerate(source_sents_no_stop):
        if sent in seen:
            scores[idx] = -1
        else:
            seen.add(sent)
    sent_order = scores.argsort()[::-1]
    rouges = [scores[i] for i in sent_order]
    return sent_order, rouges


def max_rouge_set(target, source_sents, rouge_types, target_tok_ct=None, tol=0.005):
    n = len(source_sents)
    curr_sum = ''
    curr_rouge = 0.0
    sent_order = []
    rouges = []
    metric = 'f1'
    for _ in range(n):
        _, idx, score = max_rouge_sent(
            target, source_sents, rouge_types, return_score=True, source_prefix=curr_sum,
            mask_idxs=sent_order, metric=metric
        )

        decreasing_score = score <= curr_rouge + tol
        mc = target_tok_ct is not None and len(source_sents[idx].split(' ')) + len(curr_sum.split(' ')) > target_tok_ct
        if decreasing_score or mc:
            break
        curr_rouge = score
        curr_sum += source_sents[idx] + ' '
        sent_order.append(idx)
        rouges.append(curr_rouge)
    return sent_order, rouges


def max_rouge_sent(
        target, source_sents, rouge_types, return_score=False, source_prefix='', mask_idxs=[], metric='f1'
):
    n = len(source_sents)
    predictions = [source_prefix + s for s in source_sents]
    references = [target for _ in range(n)]
    outputs = compute(
        predictions=predictions, references=references, rouge_types=rouge_types, use_aggregator=False)
    if metric == 'f1':
        scores = np.array(
            [sum([outputs[t][i].fmeasure for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    elif metric == 'recall':
        scores = np.array(
            [sum([outputs[t][i].recall for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    elif metric == 'precision':
        scores = np.array(
            [sum([outputs[t][i].precision for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    if len(mask_idxs) > 0:
        scores[mask_idxs] = float('-inf')
    max_idx = np.argmax(scores)
    max_source_sent = source_sents[max_idx]
    if return_score:
        return max_source_sent, max_idx, scores[max_idx]
    return max_source_sent, max_idx
