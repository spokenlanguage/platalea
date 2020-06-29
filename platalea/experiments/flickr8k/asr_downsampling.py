import logging
import random
from shutil import copyfile
import torch

import platalea.asr as M
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


batch_size = 8

factors = [3, 9, 27, 81, 243]
lz = len(str(abs(factors[-1])))
for ds_factor in factors:
    logging.info('Loading data')
    data = dict(
        train=D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language, args.audio_features_fn,
                                split='train', batch_size=batch_size,
                                shuffle=True, downsampling_factor=ds_factor),
        val=D.flickr8k_loader(args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language, args.audio_features_fn,
                              split='val', batch_size=batch_size,
                              shuffle=False))

    logging.info('Building model')
    net = M.SpeechTranscriber(M.get_default_config())
    run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=args.epochs)

    logging.info('Training')
    M.experiment(net, data, run_config, slt=data['train'].dataset.is_slt())
    suffix = str(ds_factor).zfill(lz)
    res_fname = 'result_{}.json'.format(suffix)
    copyfile('result.json', res_fname)
    if data['train'].dataset.is_slt():
        copy_best(res_fname, 'net_{}.best.pt'.format(ds_factor),
                  experiment_type='slt')
    else:
        copy_best(res_fname, 'net_{}.best.pt'.format(ds_factor),
                  experiment_type='asr')
