import logging
import random
import torch

import platalea.basicvq as M
import platalea.dataset as D
from platalea.experiments.config import get_argument_parser


args = get_argument_parser()
# Parsing arguments
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

bidi = True
config = dict(
    SpeechEncoder=dict(
        SpeechEncoderBottom=dict(
            conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2,
                      padding=0, bias=False),
            rnn=dict(input_size=64, hidden_size=args.hidden_size_factor,
                     num_layers=1, bidirectional=bidi, dropout=0)),
        VQEmbedding=dict(num_codebook_embeddings=1024,
                         embedding_dim=2 * args.hidden_size_factor if bidi else args.hidden_size_factor,
                         jitter=0.12),
        SpeechEncoderTop=dict(
            rnn=dict(input_size=2 * args.hidden_size_factor if bidi else args.hidden_size_factor,
                     hidden_size=args.hidden_size_factor, num_layers=3, bidirectional=bidi, dropout=0),
            att=dict(in_size=2 * args.hidden_size_factor, hidden_size=128))),
    ImageEncoder=dict(
        linear=dict(in_size=2048, out_size=2 * args.hidden_size_factor),
        norm=True),
    margin_size=0.2)


logging.info('Building model')
net = M.SpeechImage(config)
run_config = dict(max_lr=2 * 1e-4, epochs=32)

logging.info('Training')
M.experiment(net, data, run_config)
