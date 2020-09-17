import logging
import random
import torch

import platalea.basic as M
import platalea.encoders
import platalea.dataset as D
from platalea.experiments.config import args


# Parsing arguments
args.add_argument('--batch_size', default=32, type=int,
                  help='How many samples per batch to load.')

args.add_argument('--trafo_d_model', default=512, type=int,
                  help='TRANSFORMER: The dimensionality of the transformer model.')
args.add_argument('--trafo_encoder_layers', default=4, type=int,
                  help='TRANSFORMER: Number of transformer encoder layers.')
args.add_argument('--trafo_heads', default=8, type=int,
                  help='TRANSFORMER: Number of attention heads.')
args.add_argument('--trafo_feedforward_dim', default=1024, type=int,
                  help='TRANSFORMER: Dimensionality of feedforward layer at the end of the transformer layer stack.')

args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)

# Logging the arguments
logging.info('Arguments: {}'.format(args))


logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
                            args.audio_features_fn, split='train', batch_size=args.batch_size, shuffle=True,
        downsampling_factor=args.downsampling_factor),
    val=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
                          args.audio_features_fn, split='val', batch_size=args.batch_size, shuffle=False))

speech_config = {'conv': dict(in_channels=39, out_channels=64, kernel_size=6, stride=2, padding=0, bias=False),
                 'trafo': dict(d_model=args.trafo_d_model, dim_feedforward=args.trafo_feedforward_dim,
                               num_encoder_layers=args.trafo_encoder_layers, dropout=0, nhead=args.trafo_heads),
                 'upsample': dict(bias=True),
                 'att': dict(in_size=args.trafo_d_model, hidden_size=128),
                 }
speech_encoder = platalea.encoders.SpeechEncoderTransformer(speech_config)

# these must match, otherwise the loss cannot be calculated
image_encoder_out_size = args.trafo_d_model

config = dict(SpeechEncoder=speech_encoder,
              ImageEncoder=dict(linear=dict(in_size=2048, out_size=image_encoder_out_size), norm=True),
              margin_size=0.2)

logging.info('Building model')
net = M.SpeechImage(config)
run_config = dict(max_lr=2 * 1e-4, epochs=args.epochs, lr_scheduler=args.lr_scheduler,
                  d_model=args.trafo_d_model)

logging.info('Training')
M.experiment(net, data, run_config)
