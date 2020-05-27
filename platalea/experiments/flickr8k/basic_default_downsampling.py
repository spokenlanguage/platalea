import logging
import random
from shutil import copyfile
import torch

import platalea.basic as M
import platalea.dataset as D
from platalea.utils.copy_best import copy_best
from platalea.experiments.config import args


# Parsing arguments
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)
logging.basicConfig(level=logging.INFO)


factors = [3, 9, 27, 81, 243]
lz = len(str(abs(factors[-1])))
for ds_factor in factors:
    logging.info('Loading data')
    data = dict(
        train=D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta,
                                args.flickr8k_language, args.audio_features_fn,
                                split='train', batch_size=32, shuffle=True,
                                downsampling_factor=ds_factor),
        val=D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta,
                              args.flickr8k_language, args.audio_features_fn,
                              split='val', batch_size=32, shuffle=False))

    config = dict(
        SpeechEncoder=dict(
            conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2,
                      padding=0, bias=False),
            rnn=dict(input_size=64, hidden_size=1024, num_layers=4,
                     bidirectional=True, dropout=0),
            att=dict(in_size=2048, hidden_size=128)),
        ImageEncoder=dict(linear=dict(in_size=2048, out_size=2*1024),
                          norm=True),
        margin_size=0.2)

    logging.info('Building model')
    net = M.SpeechImage(config)
    run_config = dict(max_lr=2 * 1e-4, epochs=32)

    logging.info('Training')
    M.experiment(net, data, run_config)
    suffix = str(ds_factor).zfill(lz)
    res_fname = 'result_{}.json'.format(suffix)
    copyfile('result.json', res_fname)
    copy_best(res_fname, 'net_{}.best.pt'.format(ds_factor))
