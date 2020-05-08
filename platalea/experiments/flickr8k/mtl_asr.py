import logging
import random
import torch
import torch.nn as nn

import platalea.dataset as D
import platalea.mtl as M
from platalea.score import score, score_asr, score_slt
from platalea.experiments.config import args


# Parsing arguments
args.add_argument(
    '--downsampling_factor_text', default=None, type=int,
    help='factor by which the amount of available transcriptions should be \
    downsampled (affecting ASR only)')
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
        shuffle=False, downsampling_factor=args.downsampling_factor))
fd = D.Flickr8KData

if args.downsampling_factor_text:
    ds_factor_text = args.downsampling_factor_text
    step_asr = args.downsampling_factor_text
    # The downsampling factor for text is applied on top of the main
    # downsampling factor that is applied to all data
    if args.downsampling_factor:
        ds_factor_text *= args.downsampling_factor
    data_asr = dict(
        train=D.flickr8k_loader(
            split='train', batch_size=batch_size, shuffle=True,
            downsampling_factor=ds_factor_text),
        val=D.flickr8k_loader(
            split='val', batch_size=batch_size,
            downsampling_factor=ds_factor_text))
else:
    data_asr = data
    step_asr = 1

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
        emb=dict(num_embeddings=fd.vocabulary_size(),
                 embedding_dim=hidden_size),
        drop=dict(p=dropout),
        att=dict(in_size_enc=hidden_size * 2, in_size_state=hidden_size,
                 hidden_size=hidden_size),
        rnn=dict(input_size=hidden_size * 3, hidden_size=hidden_size,
                 num_layers=1, dropout=dropout),
        out=dict(in_features=hidden_size * 3,
                 out_features=fd.vocabulary_size()),
        rnn_layer_type=nn.GRU,
        max_output_length=400,  # max length for flickr annotations is 199
        sos_id=fd.get_token_id(fd.sos),
        eos_id=fd.get_token_id(fd.eos),
        pad_id=fd.get_token_id(fd.pad)),
    inverse_transform_fn=fd.get_label_encoder().inverse_transform,
    margin_size=0.2,
    lmbd=0.5)

logging.info('Building model')
net = M.MTLNetASR(config)
run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=32)

if data['train'].dataset.is_slt():
    scorer = score_slt
else:
    scorer = score_asr
tasks = [
    dict(name='SI', net=net.SpeechImage, data=data, eval=score, step=1),
    dict(name='ASR', net=net.SpeechTranscriber, data=data_asr, eval=scorer,
         step=step_asr)]

logging.info('Training')
M.experiment(net, tasks, run_config)
