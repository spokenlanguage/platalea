import logging
import random
import torch

import platalea.basic as M
import platalea.dataset as D
from platalea.experiments.config import args


# Parsing arguments
args.add_argument('--epochs', action='store', default=32, type=int,
                  help='number of epochs after which to stop training')
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)
logging.basicConfig(level=logging.INFO)


logging.info('Loading data')
data = dict(train=D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta,
                                    args.flickr8k_language, args.audio_features_fn,
                                    split='train', batch_size=32, shuffle=True),
            val=D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta,
                                  args.flickr8k_language, args.audio_features_fn,
                                  split='val', batch_size=32, shuffle=False))

config = dict(
    SpeechEncoder=dict(
        conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2,
                  padding=0, bias=False),
        rnn=dict(input_size=64, hidden_size=1024, num_layers=4,
                 bidirectional=True, dropout=0),
        att=dict(in_size=2048, hidden_size=128)),
    ImageEncoder=dict(
        linear=dict(in_size=2048, out_size=2*1024),
        norm=True),
    margin_size=0.2)

logging.info('Building model')
net = M.SpeechImage(config)
run_config = dict(max_lr=2 * 1e-4, epochs=args.epochs)

logging.info('Training')
M.experiment(net, data, run_config)
