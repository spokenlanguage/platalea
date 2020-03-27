import configargparse
import logging
import pickle
import random
import torch

import platalea.text_image as M
import platalea.dataset as D

# Parsing arguments
parser = configargparse.get_argument_parser('platalea')
parser.add_argument(
    '--seed', default=123, type=int,
    help='seed for sources of randomness (default: 123)')
config_args, _ = parser.parse_known_args()

# Setting general configuration
torch.manual_seed(config_args.seed)
random.seed(config_args.seed)
logging.basicConfig(level=logging.INFO)


batch_size = 32
hidden_size = 1024
dropout = 0.0

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(split='train', batch_size=batch_size, shuffle=True),
    val=D.flickr8k_loader(split='val', batch_size=batch_size, shuffle=False))

logging.info('Building model')
net = M.TextImage(M.get_default_config())
run_config = dict(max_lr=2 * 1e-4, epochs=32)

logging.info('Training')
M.experiment(net, data, run_config)
