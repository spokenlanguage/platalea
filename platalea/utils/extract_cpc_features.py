#!/usr/bin/env python3

"""
Extract features from a pretrained CPC model

Requires https://github.com/tuanh208/CPC_audio.git@zerospeech to work

"""

import argparse
import json
import logging
import os
from pathlib import Path
import torch

from cpc.feature_loader import buildFeature, FeatureModule, loadModel

from platalea.experiments.config import get_argument_parser


def read_args(path_args):
    print(f"Loading args from {path_args}")
    with open(path_args, 'r') as file:
        args = argparse.Namespace(**json.load(file))
    return args


def load_feature_maker_CPC(cpc_path, gru_level=-1, on_gpu=True):
    assert cpc_path[-3:] == ".pt"
    assert os.path.exists(cpc_path), \
        f"CPC path at {cpc_path} does not exist!!"

    pathConfig = os.path.join(os.path.dirname(cpc_path), "checkpoint_args.json")
    CPC_args = read_args(pathConfig)

    # Load FeatureMaker
    if gru_level is not None and gru_level > 0:
        updateConfig = argparse.Namespace(nLevelsGRU=gru_level)
    else:
        updateConfig = None

    model = loadModel([cpc_path], updateConfig=updateConfig)[0]

    feature_maker = FeatureModule(model, CPC_args.onEncoder)
    feature_maker = FeatureModule(model, False)
    feature_maker.eval()
    if on_gpu:
        feature_maker.cuda()
    return feature_maker


def cpc_feature_extraction(feature_maker, x, seq_norm=False, strict=True,
                           max_size_seq=10240):
    return buildFeature(feature_maker, x,
                        strict=strict,
                        maxSizeSeq=max_size_seq,
                        seqNorm=seq_norm)


if __name__ == '__main__':
    # Parsing command line
    doc = __doc__.strip("\n").split("\n", 1)
    args = get_argument_parser()
    args._parser.description = doc[0]
    args.enable_help()
    args.add_argument(
        'audio_dir',
        help='Folder under which the audio files to process are stored.',
        type=str)
    args.add_argument(
        'model_path',
        help='Path to the pretrained CPC model.')
    args.add_argument(
        '--audio_extension', type=str, default='wav',
        help='Extension of the audio files.')
    args.add_argument(
        '--rnn_layer', type=int, default=-1,
        help='The RNN layer that needs to be extracted. '
             'Default to -1, extracts the last RNN layer of the aggregator '
             'network.')
    args.parse()

    logging.info('Loading CPC model')
    feature_maker_X = load_feature_maker_CPC(args.model_path,
                                             gru_level=args.rnn_layer)
    for cap in Path(args.audio_dir).glob(f'*.{args.audio_extension}'):
        logging.info(f'Processing {cap}')
        features = cpc_feature_extraction(feature_maker_X, cap)[0]
        torch.save(features, open(cap.with_suffix('.pt'), 'wb'))
