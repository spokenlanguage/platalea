import logging
import random
import torch
import torch.nn as nn

import platalea.dataset as D
import platalea.mtl as M
from platalea.score import score, score_asr, score_slt
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
hidden_size = args.hidden_size_factor * 3 // 4
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
fd = D.Flickr8KData

num_tokens = len(D.tokenizer.classes_)
config = dict(
    SharedEncoder=dict(
        conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2,
                  padding=0, bias=False),
        rnn=dict(input_size=64, hidden_size=hidden_size, num_layers=4,
                 bidirectional=True, dropout=dropout),
        rnn_layer_type=nn.GRU),
    SpeechEncoderTopSI=dict(
        rnn=dict(input_size=hidden_size * 2, hidden_size=hidden_size,
                 num_layers=1, bidirectional=True, dropout=dropout),
        att=dict(in_size=hidden_size * 2, hidden_size=128),
        rnn_layer_type=nn.GRU),
    SpeechEncoderTopASR=dict(
        rnn=dict(input_size=hidden_size * 2, hidden_size=hidden_size,
                 num_layers=1, bidirectional=True, dropout=dropout),
        rnn_layer_type=nn.GRU),
    ImageEncoder=dict(
        linear=dict(in_size=2048, out_size=hidden_size * 2),
        norm=True),
    TextDecoder=dict(
        emb=dict(num_embeddings=num_tokens,
                 embedding_dim=hidden_size),
        drop=dict(p=dropout),
        att=dict(in_size_enc=hidden_size * 2, in_size_state=hidden_size,
                 hidden_size=hidden_size),
        rnn=dict(input_size=hidden_size * 3, hidden_size=hidden_size,
                 num_layers=1, dropout=dropout),
        out=dict(in_features=hidden_size * 3,
                 out_features=num_tokens),
        rnn_layer_type=nn.GRU,
        max_output_length=400),  # max length for flickr annotations is 199
    margin_size=0.2,
    lmbd=0.5)

logging.info('Building model')
net = M.MTLNetASR(config)
run_config = dict(max_norm=2.0, max_lr=args.cyclic_lr_max, min_lr=args.cyclic_lr_min, epochs=args.epochs,
                  l2_regularization=args.l2_regularization,
                  loss_logging_interval=args.loss_logging_interval,
                  validation_interval=args.validation_interval,
                  opt=args.optimizer
                  )

if data['train'].dataset.is_slt():
    scorer = score_slt
else:
    scorer = score_asr
tasks = [dict(name='SI', net=net.SpeechImage, data=data, eval=score),
         dict(name='ASR', net=net.SpeechTranscriber, data=data, eval=scorer)]

logging.info('Training')
result = M.experiment(net, tasks, run_config)
