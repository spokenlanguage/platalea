import json
import logging
import random
import torch

import platalea.dataset as D
import platalea.score
from platalea.experiments.config import args


# Parsing arguments
args.add_argument('path', metavar='path', help='Model\'s path')
args.add_argument('-b', help='Use beam decoding', dest='use_beam_decoding',
                  action='store_true', default=False)
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)


batch_size = 16

logging.info('Loading data')
data = dict(val=D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta,
                                  args.flickr8k_language, args.audio_features_fn,
                                  split='val', batch_size=batch_size,
                                  shuffle=False))

logging.info('Loading model')
net = torch.load(args.path)
logging.info('Evaluating')
with torch.no_grad():
    net.eval()
    if args.use_beam_decoding:
        result = platalea.score.score_asr(net, data['val'].dataset,
                                          beam_size=10)
    else:
        result = platalea.score.score_asr(net, data['val'].dataset)
print(json.dumps(result))
