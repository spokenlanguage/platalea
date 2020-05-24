import logging
import random
import torch
import torch.nn as nn

import platalea.dataset as D
import platalea.mtl as M
from platalea.score import score, score_speech_text
from platalea.experiments.config import args


# Parsing arguments
args.add_argument(
    '--downsampling_factor_text', default=None, type=float,
    help='factor by which the amount of available transcriptions should be \
    downsampled (affecting speech-text retrieval only)')
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)
logging.basicConfig(level=logging.INFO)

# Logging the arguments
logging.info('Arguments: {}'.format(args))


batch_size = 8
hidden_size = 1024
dropout = 0.0

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

if args.downsampling_factor_text:
    ds_factor_text = args.downsampling_factor_text
    step_st = args.downsampling_factor_text
    # The downsampling factor for text is applied on top of the main
    # downsampling factor that is applied to all data
    if args.downsampling_factor:
        ds_factor_text *= args.downsampling_factor
    data_st = dict(
        train=D.flickr8k_loader(
            split='train', batch_size=batch_size, shuffle=True,
            downsampling_factor=ds_factor_text),
        val=D.flickr8k_loader(split='val', batch_size=batch_size))
else:
    data_st = data
    step_st = 1

config = dict(
    SharedEncoder=dict(
        conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2,
                  padding=0, bias=False),
        rnn=dict(input_size=64, hidden_size=hidden_size, num_layers=2,
                 bidirectional=True, dropout=dropout),
        rnn_layer_type=nn.GRU),
    SpeechEncoderTopSI=dict(
        rnn=dict(input_size=hidden_size * 2, hidden_size=hidden_size,
                 num_layers=2, bidirectional=True, dropout=dropout),
        att=dict(in_size=hidden_size * 2, hidden_size=128),
        rnn_layer_type=nn.GRU),
    SpeechEncoderTopST=dict(
        att=dict(in_size=hidden_size * 2, hidden_size=128)),
    ImageEncoder=dict(
        linear=dict(in_size=2048, out_size=hidden_size * 2),
        norm=True),
    TextEncoder=dict(
        emb=dict(num_embeddings=D.Flickr8KData.vocabulary_size(),
                 embedding_dim=128),
        rnn=dict(input_size=128, hidden_size=1024, num_layers=1,
                 bidirectional=True, dropout=0),
        att=dict(in_size=1024 * 2, hidden_size=128)),
    margin_size=0.2,
    lmbd=0.5)

logging.info('Building model')
net = M.MTLNetSpeechText(config)
run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=32)

tasks = [
    dict(name='SI', net=net.SpeechImage, data=data, eval=score, step=1),
    dict(name='ST', net=net.SpeechText, data=data_st, eval=score_speech_text,
         step=step_st)]

logging.info('Training')
M.experiment(net, tasks, run_config)
