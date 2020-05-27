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


batch_size = 8

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(split='train', batch_size=batch_size, shuffle=True),
    val=D.flickr8k_loader(split='val', batch_size=batch_size, shuffle=False))

logging.info('Building model')
net = M.SpeechTranscriber(M.get_default_config())
run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=32)

logging.info('Training')
M.experiment(net, data, run_config, slt=data['train'].dataset.is_slt())
