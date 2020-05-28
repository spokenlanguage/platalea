#!/usr/bin/env python3

import json
import logging
import torch

import platalea.dataset as D
import platalea.score
from platalea.experiments.config import args


# Parsing arguments
args.add_argument('path', metavar='path', help='Model\'s path')
args.add_argument('-b', help='Use beam decoding', dest='use_beam_decoding',
                  action='store_true', default=False)
args.add_argument('-t', help='Evaluate on test set', dest='use_test_set',
                  action='store_true', default=False)
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)
logging.basicConfig(level=logging.INFO)

batch_size = 16


logging.info('Loading data')
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

logging.info('Loading model')
net = torch.load(args.path)
logging.info('Evaluating')
with torch.no_grad():
    net.eval()
    if data.dataset.is_slt():
        score_fn = platalea.score.score_slt
    else:
        score_fn = platalea.score.score_asr
    if args.use_beam_decoding:
        result = score_fn(net, data.dataset, beam_size=10)
    else:
        result = score_fn(net, data.dataset)
print(json.dumps(result))
