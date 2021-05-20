import logging
import random
import torch

import platalea.asr as M
import platalea.dataset as D
from platalea.experiments.config import get_argument_parser


args = get_argument_parser()
# Parsing arguments
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)

# Logging the arguments
logging.info('Arguments: {}'.format(args))


batch_size = 8

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
        args.audio_features_fn, split='train', batch_size=batch_size,
        shuffle=True, downsampling_factor=args.downsampling_factor),
    val=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
        args.audio_features_fn, split='val', batch_size=batch_size,
        shuffle=False))

logging.info('Building model')
net = M.SpeechTranscriber(M.get_default_config(hidden_size_factor=args.hidden_size_factor))
run_config = dict(max_norm=2.0, max_lr=args.cyclic_lr_max, min_lr=args.cyclic_lr_min, epochs=args.epochs,
                  l2_regularization=args.l2_regularization,
                  loss_logging_interval=args.loss_logging_interval,
                  validation_interval=args.validation_interval,
                  opt=args.optimizer
                  )

logging.info('Training')
result = M.experiment(net, data, run_config, slt=data['train'].dataset.is_slt())
