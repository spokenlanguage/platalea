import logging
import random
from shutil import copyfile
import torch
import torch.nn as nn

import platalea.dataset as D
import platalea.mtl as M
from platalea.score import score, score_asr, score_slt
from utils.copy_best import copy_best
from platalea.experiments.config import args


# Parsing arguments
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)


batch_size = 8
hidden_size = 1024
dropout = 0.0

factors = [3, 9, 27, 81, 243]
lz = len(str(abs(factors[-1])))
for ds_factor in factors:
    logging.info('Loading data')
    data = dict(
        train=D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta,
                                args.flickr8k_language, args.audio_features_fn,
                                split='train', batch_size=batch_size,
                                shuffle=True, downsampling_factor=ds_factor),
        val=D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta,
                              args.flickr8k_language, args.audio_features_fn,
                              split='val', batch_size=batch_size,
                              shuffle=False))
    fd = D.Flickr8KData

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
    run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=args.epochs)

    if data['train'].dataset.is_slt():
        scorer = score_slt
    else:
        scorer = score_asr
    tasks = [
        dict(name='SI', net=net.SpeechImage, data=data, eval=score),
        dict(name='ASR', net=net.SpeechTranscriber, data=data, eval=scorer)]

    logging.info('Training')
    M.experiment(net, tasks, run_config)
    suffix = str(ds_factor).zfill(lz)
    res_fname = 'result_{}.json'.format(suffix)
    copyfile('result.json', res_fname)
    copy_best(res_fname, 'net_{}.best.pt'.format(ds_factor),
              experiment_type='mtl')
