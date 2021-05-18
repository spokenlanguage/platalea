#!/usr/bin/env python3

"""
Copy best network to net.best.pt

Go through results to find the best performing epoch and copy the corresponding
network to net.best.pt.
"""


import argparse
import numpy as np
import pathlib
from shutil import copyfile

from platalea.utils.get_best_score import read_results, get_metric_accessor


def copy_best(exp_path=['.'], result_fname='result.json', save_fname='net.best.pt',
              experiment_type='retrieval'):
    for path in exp_path:
        root_path = pathlib.Path(path)
        res = read_results(root_path / result_fname)
        metric_accessor = get_metric_accessor(experiment_type)
        if experiment_type == 'asr':
            ibest = np.argmin([metric_accessor(r) for r in res]) + 1
        else:
            ibest = np.argmax([metric_accessor(r) for r in res]) + 1
        best_fname = 'net.{}.pt'.format(ibest)
        copyfile(root_path / best_fname, root_path / save_fname)


if __name__ == '__main__':
    # Parsing command line
    doc = __doc__.strip("\n").split("\n", 1)
    parser = argparse.ArgumentParser()
    parser.description = doc[0]
    parser.epilog = doc[1]
    parser.add_argument(
        'exp_path', help='Path to the experiment', default=['.'], nargs='*')
    parser.add_argument(
        '--result',
        help='Path to the JSON file containing the results'
        ' (default=result.json).',
        type=str, default='result.json')
    parser.add_argument(
        '--save',
        help='Path where the corresponding net should be saved'
        ' (default=net.best.pt).',
        type=str, default='net.best.pt')
    parser.add_argument(
        '--experiment_type', dest='experiment_type',
        help='Type of experiment. Determines which metric is used'
        ' (default=retrieval).',
        type=str, choices=['retrieval', 'asr', 'mtl', 'slt'],
        default='retrieval')
    args = parser.parse_args()

    copy_best(args.exp_path, args.result, args.save, args.experiment_type)
