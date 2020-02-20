#!/usr/bin/env python3

"""
Copy best network to net.best.pt

Go through results to find the best performing epoch and copy the corresponding
network to net.best.pt.
"""


import argparse
import json
import numpy as np
from shutil import copyfile


def read_results(fpath="result.json"):
    res = []
    content = open(fpath).readlines()
    for line in content:
        res.append(json.loads(line))
    return res


def get_metric_accessor(experiment_type):
    if experiment_type == 'retrieval':
        return lambda x: x['recall']['10']
    elif experiment_type == 'asr':
        return lambda x: x['wer']['WER']
    elif experiment_type == 'mtl':
        return lambda x: x['SI']['recall']['10']


def copy_best(result_fpath='result.json', save_fpath='net.best.pt',
              metric_accessor=get_metric_accessor('retrieval')):
    res = read_results(result_fpath)
    ibest = np.argmax([metric_accessor(r) for r in res]) + 1
    copyfile('net.{}.pt'.format(ibest), 'net.best.pt')


if __name__ == '__main__':
    # Parsing command line
    doc = __doc__.strip("\n").split("\n", 1)
    parser = argparse.ArgumentParser(
        description=doc[0], epilog=doc[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--result', help='Path to the JSON file containing the results.',
        type=str, action='store', default='result.json')
    parser.add_argument(
        '--save', help='Path where the corresponding net should be saved.',
        type=str, action='store', default='net.best.pt')
    parser.add_argument(
        '--experiment-type', dest='experiment_type',
        help='Type of experiment. Determines which metric is used.',
        type=str, action='store', choices=['retrieval', 'asr', 'mtl'],
        default='retrieval')
    args = parser.parse_args()

    metric_accessor = get_metric_accessor(args.experiment_type)
    copy_best(args.result, args.save, metric_accessor)
