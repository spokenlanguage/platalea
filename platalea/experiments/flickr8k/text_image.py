import logging
import random
import torch

import platalea.text_image as M
import platalea.dataset as D
from platalea.experiments.config import get_argument_parser


args = get_argument_parser()
# Parsing arguments
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)


batch_size = 32
hidden_size = 1024
dropout = 0.0

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta,
                            args.flickr8k_language, args.audio_features_fn,
                            split='train', batch_size=batch_size, shuffle=True),
    val=D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta,
                          args.flickr8k_language, args.audio_features_fn,
                          split='val', batch_size=batch_size, shuffle=False))

logging.info('Building model')
net = M.TextImage(M.get_default_config())
run_config = dict(max_lr=args.cyclic_lr_max, min_lr=args.cyclic_lr_min, epochs=args.epochs,
                  l2_regularization=args.l2_regularization,
                  loss_logging_interval=args.loss_logging_interval,
                  validation_interval=args.validation_interval,
                  opt=args.optimizer
                  )

logging.info('Training')
result = M.experiment(net, data, run_config)
