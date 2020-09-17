#!/usr/bin/env python3

"""
Returns the best score

Go through results to find the best performing epoch and returns the
corresponding score.
"""


import argparse
import json
import numpy as np


def read_results(fpath="result.json"):
    res = []
    content = open(fpath).readlines()
    for line in content:
        try:
            res.append(json.loads(line))
        except json.JSONDecodeError as e:
            print('Error reading {}'.format(fpath))
            raise e
    return res


def get_metric_accessor(experiment_type):
    if experiment_type == 'retrieval':
        return lambda x: x['recall']['10']
    elif experiment_type == 'asr':
        return lambda x: x['wer']['WER']
    elif experiment_type == 'slt':
        return lambda x: x['bleu']
    elif experiment_type == 'mtl':
        return lambda x: x['SI']['recall']['10']


def get_best_score(result_fpath='result.json', experiment_type='retrieval'):
    res = read_results(result_fpath)
    metric_accessor = get_metric_accessor(experiment_type)
    if experiment_type == 'asr':
        best = np.min([metric_accessor(r) for r in res])
    else:
        best = np.max([metric_accessor(r) for r in res])
    return best


if __name__ == '__main__':
    # Parsing command line
    doc = __doc__.strip("\n").split("\n", 1)
    parser = argparse.ArgumentParser(description=doc[0], epilog=doc[1])
    parser.add_argument(
        'res_fpath', help='Path to the JSON file(s) containing the results.',
        type=str, action='store', default='result.json', nargs='+')
    parser.add_argument(
        '--separator',
        help='Separator used when several result files are given.',
        type=str, action='store', default=',')
    parser.add_argument(
        '--experiment_type', dest='experiment_type',
        help='Type of experiment. Determines which metric is used.',
        type=str, action='store', choices=['retrieval', 'asr', 'mtl', 'slt'],
        default='retrieval')
    args = parser.parse_args()

    scores = [str(get_best_score(f, args.experiment_type)) for f in args.res_fpath]
    print(args.separator.join(scores))
