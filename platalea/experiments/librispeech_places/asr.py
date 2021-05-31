import logging
import pickle
import random
import torch
import torch.nn as nn

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


batch_size = 8
hidden_size = 1024 * 3 // 4
dropout = 0.0

logging.info('Loading data')
data = dict(
    train=D.librispeech_loader(args.librispeech_root, args.librispeech_meta,
                               args.audio_features_fn,
                               split='train', batch_size=batch_size,
                               shuffle=True, downsampling_factor=10),
    val=D.librispeech_loader(args.librispeech_root, args.librispeech_meta,
                             args.audio_features_fn,
                             split='val', batch_size=batch_size))
D.LibriSpeechData.init_vocabulary(data['train'].dataset)

# Saving config
pickle.dump(data['train'].dataset.get_config(),
            open('config.pkl', 'wb'))

num_tokens = len(D.tokenizer.classes_)
config = dict(
    SpeechEncoder=dict(
        conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2,
                  padding=0, bias=False),
        rnn=dict(input_size=64, hidden_size=hidden_size, num_layers=5,
                 bidirectional=True, dropout=dropout),
        rnn_layer_type=nn.GRU),
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
        max_output_length=400))  # max length for flickr annotations is 199

logging.info('Building model')
net = M.SpeechTranscriber(config)
run_config = dict(max_norm=2.0, max_lr=args.cyclic_lr_max, min_lr=args.cyclic_lr_min, epochs=args.epochs,
                  l2_regularization=args.l2_regularization,
                  loss_logging_interval=args.loss_logging_interval,
                  validation_interval=args.validation_interval,
                  opt=args.optimizer
                  )

logging.info('Training')
M.experiment(net, data, run_config)
