import logging
import random
import torch

import platalea.asr as M
import platalea.dataset as D
from platalea.experiments.config import args


# Parsing arguments
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)
logging.basicConfig(level=logging.INFO)

# Logging the arguments
logging.info('Arguments: {}'.format(args))


batch_size = 8

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
        args.audio_features_fn, split='train', batch_size=batch_size,
        shuffle=True, downsampling_factor=args.downsampling_factor)
    val=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
        args.audio_features_fn, split='val', batch_size=batch_size,
        shuffle=False)

logging.info('Building model')
net = M.SpeechTranscriber(M.get_default_config())
run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=32)

logging.info('Training')
M.experiment(net, data, run_config, slt=data['train'].dataset.is_slt())
