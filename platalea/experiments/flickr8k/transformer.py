import logging
import random
import torch

import platalea.basic as M
import platalea.encoders
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

# Logging the arguments
logging.info('Arguments: {}'.format(args))


logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
        args.audio_features_fn, split='train', batch_size=32, shuffle=True,
        downsampling_factor=args.downsampling_factor),
    val=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
        args.audio_features_fn, split='val', batch_size=32, shuffle=False))

trafo_d_model = 512
speech_config = {'conv': dict(in_channels=39, out_channels=64, kernel_size=6, stride=2, padding=0, bias=False),
                 'trafo': dict(d_model=trafo_d_model, dim_feedforward=1024, num_encoder_layers=4, dropout=0, nhead=8),
                 'upsample': dict(bias=True),
                 'att': dict(in_size=trafo_d_model, hidden_size=128),
                 }
speech_encoder = platalea.encoders.SpeechEncoderTransformer(speech_config)

# these must match, otherwise the loss cannot be calculated
image_encoder_out_size = trafo_d_model

config = dict(SpeechEncoder=speech_encoder,
              ImageEncoder=dict(linear=dict(in_size=2048, out_size=image_encoder_out_size), norm=True),
              margin_size=0.2)

logging.info('Building model')
net = M.SpeechImage(config)
run_config = dict(max_lr=2 * 1e-4, epochs=args.epochs)

logging.info('Training')
M.experiment(net, data, run_config)
