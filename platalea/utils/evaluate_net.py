#!/usr/bin/env python3
import json
import logging
import torch

import platalea.dataset as D
import platalea.score

from platalea.asr import SpeechTranscriber
from platalea.basic import SpeechImage
from platalea.mtl import MTLNetASR, MTLNetSpeechText
from platalea.speech_text import SpeechText
from platalea.text_image import TextImage
from platalea.experiments.config import get_argument_parser


def get_score_fn_speech_transcriber(is_slt, use_beam_decoding):
    if is_slt:
        score_fn = platalea.score.score_slt
    else:
        score_fn = platalea.score.score_asr
    if use_beam_decoding:
        score_fn = lambda x, y: score_fn(x, y, beam_size=10)
    return score_fn


def get_evaluation_tasks(net):
    if type(net) == SpeechImage:
        return [dict(net=net, score_fn=platalea.score.score)]
    elif type(net) == TextImage:
        return [dict(net=net, score_fn=platalea.score.score_text_image)]
    elif type(net) == SpeechText:
        return [dict(net=net, score_fn=platalea.score.score_speech_text)]
    elif type(net) == SpeechTranscriber:
        score_fn = get_score_fn_speech_transcriber(data.dataset.is_slt(),
                                                   args.use_beam_decoding)
        return [dict(net=net, score_fn=score_fn)]
    elif type(net) == MTLNetASR:
        score_fn = get_score_fn_speech_transcriber(data.dataset.is_slt(),
                                                   args.use_beam_decoding)
        return [dict(name='SI', net=net.SpeechImage,
                     score_fn=platalea.score.score),
                dict(name='ASR', net=net.SpeechTranscriber,
                     score_fn=score_fn)]
    elif type(net) == MTLNetSpeechText:
        return [dict(name='SI', net=net.SpeechImage,
                     score_fn=platalea.score.score),
                dict(name='ST', net=net.SpeechText,
                     score_fn=platalea.score.score_speech_text)]


# Parsing arguments
args = get_argument_parser()
args.add_argument('paths', metavar='paths', nargs='+', help='Model\'s path')
args.add_argument('-b', help='Use beam decoding (for ASR and SLT experiments)',
                  dest='use_beam_decoding', action='store_true', default=False)
args.add_argument('-t', help='Evaluate on test set', dest='use_test_set',
                  action='store_true', default=False)
args.add_argument('--dataset_name', help='Name of the dataset to use for evaluation.',
                  type=str, choices=['flickr8k', 'spokencoco'],
                  default='flickr8k')
args.enable_help()
args.parse()

batch_size = 16

logging.basicConfig(level=logging.INFO)


logging.info('Loading data')
if args.dataset_name == 'flickr8k':
    if args.use_test_set:
        data = D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta,
                                 args.flickr8k_language, args.audio_features_fn,
                                 split='test', batch_size=batch_size,
                                 shuffle=False)
    else:
        data = D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta,
                                 args.flickr8k_language, args.audio_features_fn,
                                 split='val', batch_size=batch_size,
                                 shuffle=False)
elif args.dataset_name == "spokencoco":
    if args.use_test_set:
        data = D.spokencoco_loader(args.spokencoco_root, args.spokencoco_meta,
                                   args.audio_features_fn, split='test',
                                   split_scheme=args.spokencoco_split_scheme,
                                   batch_size=batch_size, shuffle=False,
                                   debug=args.debug)
    else:
        data = D.spokencoco_loader(args.spokencoco_root, args.spokencoco_meta,
                                   args.audio_features_fn, split='val',
                                   split_scheme=args.spokencoco_split_scheme,
                                   batch_size=batch_size, shuffle=False,
                                   debug=args.debug)

out = []
for p in args.paths:
    out.append(p)
    logging.info(f'Loading model {p}')
    net = torch.load(p)
    logging.info('Evaluating')
    with torch.no_grad():
        tasks = get_evaluation_tasks(net)
        net.eval()
        results = {}
        for t in tasks:
            r = t['score_fn'](t['net'], data.dataset)
            if 'name' in t:
                results[t['name']] = r
            else:
                results.update(r)
    s = json.dumps(results)
    out.append(s)
    print(s)
print('\n'.join(out))
